"""Runner CLI: one isolated headless Claude Code invocation per benchmark case.

Each case spawns its own ``claude -p`` (which spawns its own MCP-server child for
the chemistree arm), so cases do not share context and every token is attributed to
that case. The agent must end with a ``FINAL_SMILES:`` line; the runner parses it,
captures the run's token usage, and appends one JSON row per case.

Usage::

    python -m benchmarks.run --items benchmarks/sample_2d.jsonl --arm chemistree
    python -m benchmarks.run --items benchmarks/sample_2d.jsonl --arm naked --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from rdkit import Chem

from benchmarks import arms, metrics


def load_items(path: Path) -> list[dict]:
    """Read a JSONL file of cases.

    Args:
        path: Path to a file with one JSON object per line. Each needs ``id``,
            ``smiles``, and ``instruction``; ``gold`` is optional.

    Returns:
        The parsed cases, in file order.
    """
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_prompt(arm: arms.Arm, item: dict) -> str:
    """The task prompt for one case, tailored to whether the arm has the tools.

    Args:
        arm: The arm being run.
        item: The case, with ``smiles`` and ``instruction``.

    Returns:
        The prompt to pass to ``claude -p``.
    """
    smiles, instruction = item["smiles"], item["instruction"]
    contract = "End your reply with a line exactly of the form:\nFINAL_SMILES: <smiles>"
    if arm.uses_chemistree:
        return (
            "Edit a molecule with the chemistree tools.\n"
            f'1. Call bind with molecule "{smiles}".\n'
            f"2. {instruction}\n"
            "3. Call the smiles tool to read the exact canonical SMILES of the "
            "result.\n"
            f"{contract}\nUse the SMILES the smiles tool returned."
        )
    return (
        "Edit this molecule and give the result as SMILES.\n"
        f"SMILES: {smiles}\n"
        f"Task: {instruction}\n"
        f"{contract}"
    )


def build_command(
    arm: arms.Arm, prompt: str, model: str, mcp_config: Path | None
) -> list[str]:
    """The ``claude`` argv for one case.

    Args:
        arm: The arm being run.
        prompt: The task prompt.
        model: The Claude model alias.
        mcp_config: Path to the arm's ``--mcp-config`` file, or None.

    Returns:
        The argument list to spawn.
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model]
    if mcp_config is not None:
        cmd += ["--mcp-config", str(mcp_config), "--strict-mcp-config"]
    if arm.allowed_tools:
        cmd += ["--allowedTools", arm.allowed_tools]
    if arm.disallowed_tools:
        cmd += ["--disallowedTools", arm.disallowed_tools]
    return cmd


def canonical(smiles: str | None) -> str | None:
    """A canonical SMILES, or None when it is missing or unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else str(Chem.MolToSmiles(mol))


def run_case(
    arm: arms.Arm, item: dict, model: str, timeout: int, workdir: Path
) -> dict:
    """Run one case and return its result row.

    Args:
        arm: The arm to run.
        item: The case.
        model: The Claude model alias.
        timeout: Seconds before the case is abandoned.
        workdir: Directory for the per-case mcp-config and trace files.

    Returns:
        A result row: ids, status, metrics, and the final molecule.
    """
    mcp_config = None
    trace_path = None
    if arm.uses_chemistree:
        trace_path = workdir / f"trace_{item['id']}.jsonl"
        mcp_config = workdir / f"mcp_{item['id']}.json"
        mcp_config.write_text(
            json.dumps(arms.chemistree_mcp_config(arm.tools_profile, str(trace_path)))
        )
    command = build_command(arm, build_prompt(arm, item), model, mcp_config)

    row: dict = {"id": item["id"], "arm": arm.name, "model": model}
    start = time.time()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        row.update(status="timeout", wall_s=timeout)
        return row
    row["wall_s"] = round(time.time() - start, 1)

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        row.update(status="no_result", stderr=proc.stderr[-500:])
        return row

    row.update(metrics.parse_result(result))
    final = metrics.parse_final_smiles(result.get("result", ""))
    row["final_smiles"] = final
    row["status"] = "ok" if final else "no_final_smiles"
    if trace_path is not None and trace_path.exists():
        row["trace"] = str(trace_path)
    if "gold" in item:  # provisional match; the real scorer is score_2d.py
        row["gold"] = item["gold"]
        row["correct"] = canonical(final) is not None and canonical(final) == canonical(
            item["gold"]
        )
    return row


def main() -> None:
    """Parse arguments and run every case of one arm, writing a result row each."""
    parser = argparse.ArgumentParser(description="Run a benchmark arm over cases.")
    parser.add_argument("--items", required=True, help="JSONL file of cases.")
    parser.add_argument(
        "--arm", choices=sorted(arms.ARMS), default="chemistree", help="Which arm."
    )
    parser.add_argument("--model", default="haiku")
    parser.add_argument(
        "--out", default="benchmarks/results/run.jsonl", help="Result JSONL path."
    )
    parser.add_argument("--limit", type=int, help="Run only the first N cases.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-case seconds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each case's command without running it.",
    )
    args = parser.parse_args()

    arm = arms.ARMS[args.arm]
    items = load_items(Path(args.items))
    if args.limit is not None:
        items = items[: args.limit]

    workdir = Path(tempfile.mkdtemp(prefix="chemistree-bench-"))
    if args.dry_run:
        for item in items:
            print(
                json.dumps(
                    build_command(arm, build_prompt(arm, item), args.model, None)
                )
            )
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for item in items:
            row = run_case(arm, item, args.model, args.timeout, workdir)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                f"[{row['status']}] {row['id']} ({arm.name}) "
                f"final={row.get('final_smiles')} "
                f"footprint={row.get('context_footprint')} "
                f"cost=${row.get('cost_usd')}"
            )


if __name__ == "__main__":
    main()
