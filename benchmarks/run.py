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
import os
import subprocess
import tempfile
import time
from pathlib import Path

from rdkit import Chem

from benchmarks import arms, metrics, oracle_smina


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


_CONTRACT = "End your reply with a line exactly of the form:\nFINAL_SMILES: <smiles>"


def is_3d(item: dict) -> bool:
    """True for a 3D optimization case (carries a ligand file + receptor)."""
    return "ligand" in item and "receptor" in item


def ligand_smiles(item: dict) -> str:
    """The crystal ligand's canonical SMILES, for the arms that get text only."""
    return str(Chem.MolToSmiles(Chem.MolFromMolFile(item["ligand"])))


def build_prompt(arm: arms.Arm, item: dict, pose_out: Path | None = None) -> str:
    """The task prompt for one case, tailored to the arm and the track.

    Args:
        arm: The arm being run.
        item: The case.
        pose_out: Where the chemistree arm writes its final 3D pose (3D only).

    Returns:
        The prompt to pass to ``claude -p``.
    """
    if is_3d(item):
        return _prompt_3d(arm, item, pose_out)
    return _prompt_2d(arm, item)


def _prompt_2d(arm: arms.Arm, item: dict) -> str:
    """The 2D editing prompt: bind a SMILES (chemistree) or edit it as text."""
    smiles, instruction = item["smiles"], item["instruction"]
    if arm.uses_chemistree:
        return (
            "Edit a molecule with the chemistree tools.\n"
            f'1. Call bind with molecule "{smiles}".\n'
            f"2. {instruction}\n"
            "3. Call the smiles tool to read the exact canonical SMILES of the "
            "result.\n"
            f"{_CONTRACT}\nUse the SMILES the smiles tool returned."
        )
    return (
        "Edit this molecule and give the result as SMILES.\n"
        f"SMILES: {smiles}\n"
        f"Task: {instruction}\n"
        f"{_CONTRACT}"
    )


def _prompt_3d(arm: arms.Arm, item: dict, pose_out: Path | None) -> str:
    """The 3D optimization prompt, tailored to each arm's tools."""
    instruction = item["instruction"]
    if arm.uses_chemistree:
        return (
            "Lower a ligand's predicted binding affinity with the chemistree tools.\n"
            f'1. Call bind with molecule "{item["ligand"]}" and receptor '
            f'"{item["receptor"]}". The listing shows the Predicted affinity '
            "(Vinardo); lower (more negative) is better.\n"
            f"2. {instruction} Use contacts and distance to read the pocket, clashes "
            "to check strain, and minimize a group after an edit that moves atoms; "
            "re-check the affinity.\n"
            f'3. Call write_pose with path "{pose_out}" to save your final 3D pose.\n'
            "4. Call the smiles tool to read the final canonical SMILES.\n"
            f"{_CONTRACT}"
        )
    if arm.uses_chemistree is False and arm.allowed_tools == "Bash":  # generalist
        smina = os.environ.get("SMINA_BIN", "smina")
        return (
            "Lower a ligand's predicted binding affinity to a receptor. You have "
            "Bash with rdkit (python) and smina.\n"
            f"Ligand SDF: {item['ligand']}\nReceptor: {item['receptor']}\n"
            f"Score a candidate with:\n  {smina} --receptor {item['receptor']} "
            f"--ligand <candidate.sdf> --autobox_ligand {item['ligand']} "
            "--scoring vinardo --score_only\n"
            f"Task: {instruction}\n"
            f"{_CONTRACT}"
        )
    return (  # naked
        "A ligand binds a protein target. Propose an analog with better predicted "
        "binding affinity by editing its structure.\n"
        f"SMILES: {ligand_smiles(item)}\n"
        f"Task: {instruction}\n"
        f"{_CONTRACT}"
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
    pose_out = None
    if is_3d(item) and arm.uses_chemistree:
        pose_out = workdir / f"pose_{item['id']}.sdf"
    command = build_command(arm, build_prompt(arm, item, pose_out), model, mcp_config)

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
    if is_3d(item):
        row.update(_score_3d(arm, item, final, pose_out, workdir))
    elif "gold" in item:  # provisional match; the real scorer is score_2d.py
        row["gold"] = item["gold"]
        row["correct"] = canonical(final) is not None and canonical(final) == canonical(
            item["gold"]
        )
    return row


def _score_3d(
    arm: arms.Arm, item: dict, final: str | None, pose_out: Path | None, workdir: Path
) -> dict:
    """Score a 3D case with smina: redock (Column 1) and, for arm C, the pose (2).

    Args:
        arm: The arm being scored.
        item: The 3D case (ligand, receptor).
        final: The arm's final SMILES, or None.
        pose_out: The chemistree arm's written pose SDF, or None.
        workdir: Directory for smina's scratch files.

    Returns:
        Scoring fields to merge onto the result row (None where smina had nothing).
    """
    receptor, ligand = item["receptor"], item["ligand"]
    baseline = oracle_smina.redock(ligand_smiles(item), receptor, ligand, workdir)
    final_aff = oracle_smina.redock(final, receptor, ligand, workdir) if final else None
    out: dict = {"redock_baseline": baseline, "redock_final": final_aff}
    if baseline is not None and final_aff is not None:
        out["redock_delta"] = round(final_aff - baseline, 2)
        out["improved"] = final_aff < baseline
    if arm.uses_chemistree and pose_out is not None and pose_out.exists():
        out["pose_final"] = oracle_smina.score_pose(str(pose_out), receptor)
        out["pose_baseline"] = oracle_smina.score_pose(ligand, receptor)
    return out


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
            extra = (
                f"Δaffinity={row.get('redock_delta')}"
                if "redock_delta" in row
                else f"correct={row.get('correct')}"
            )
            print(
                f"[{row['status']}] {row['id']} ({arm.name}) {extra} "
                f"footprint={row.get('context_footprint')} "
                f"cost=${row.get('cost_usd')}"
            )


if __name__ == "__main__":
    main()
