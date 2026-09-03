"""Runner CLI: one isolated headless Claude Code invocation per benchmark case.

Each case spawns its own ``claude -p`` (which spawns its own MCP-server child for the
chemistree arm), so cases never share context and tokens are attributed per case.

Fairness: the **task prompt** (the ``-p`` text) is word-for-word **identical** across
arms; only the **system prompt** (``--append-system-prompt``) differs, carrying each
arm's access mechanics plus the medchem guidance (applied on every track, matching what
the app injects; ``--no-guidance`` drops it). Both prompts are recorded on each row so
parity is auditable. Each case runs in an isolated directory
(the agent's cwd) holding only the staged inputs, so a shell-capable arm cannot read the
crystal ligand (the answer) or the case files (the answer key).

Three case types, routed by their fields:
- **edit2d** (Track A): ``smiles`` + ``gold``; FINAL_SMILES, canonical match.
- **decorate** (Track B1): ``target`` + ``receptor`` — strip to a Murcko scaffold seed;
  FINAL_SMILES, redock Δ + recovery.
- **probe** (Track B2): ``ligand`` + ``receptor`` + ``question`` + ``answer`` — the full
  posed ligand; FINAL_ANSWER, answer match.

Usage::

    python -m benchmarks.run --items benchmarks/cases/abl1_2d.jsonl --arm naked
    python -m benchmarks.run --items benchmarks/cases/abl1_3d_decoration.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from benchmarks import arms, metrics, oracle_smina
from benchmarks.tasks import strip_murcko
from chemistree.mcp import guidance

_SMILES_CONTRACT = (
    "End your reply with a line exactly of the form:\nFINAL_SMILES: <smiles>"
)
_ANSWER_CONTRACT = (
    "End your reply with a line exactly of the form:\nFINAL_ANSWER: <answer>"
)
_GENERALIST_GUIDANCE = (
    (Path(__file__).parent / "prompts" / "generalist.md").read_text().strip()
)


def load_items(path: Path) -> list[dict]:
    """Read a JSONL file of cases (one JSON object per line, in file order)."""
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def case_type(item: dict) -> str:
    """The case type: ``decorate`` (B1), ``probe`` (B2), or ``edit2d`` (Track A)."""
    if "target" in item:
        return "decorate"
    if "question" in item:
        return "probe"
    return "edit2d"


# --- chemistry helpers ---------------------------------------------------------------


def target_smiles(item: dict) -> str:
    """The crystal ligand's canonical SMILES (a decorate case's recovery target)."""
    return str(Chem.MolToSmiles(Chem.MolFromMolFile(item["target"])))


def ligand_smiles(item: dict) -> str:
    """The full posed ligand's canonical SMILES (a probe case's molecule)."""
    return str(Chem.MolToSmiles(Chem.MolFromMolFile(item["ligand"])))


def canonical(smiles: str | None) -> str | None:
    """A canonical SMILES, or None when it is missing or unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else str(Chem.MolToSmiles(mol))


def tanimoto(smiles: str | None, ref_smiles: str) -> float | None:
    """ECFP4 Tanimoto similarity to a reference, or None if unparseable."""
    query = Chem.MolFromSmiles(smiles) if smiles else None
    ref = Chem.MolFromSmiles(ref_smiles)
    if query is None or ref is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect
    return float(
        round(DataStructs.TanimotoSimilarity(fp(query, 2, 2048), fp(ref, 2, 2048)), 3)
    )


def answer_matches(expected: str, got: str | None) -> bool:
    """Whether a probe answer matches, tolerant of case and SMILES canonicalization."""
    if got is None:
        return False
    if expected.strip().lower() == got.strip().lower():
        return True
    exp_smiles, got_smiles = canonical(expected), canonical(got)
    return exp_smiles is not None and exp_smiles == got_smiles


# --- prompts -------------------------------------------------------------------------


def build_prompt(item: dict, display_smiles: str) -> str:
    """The task prompt for a case — identical across arms (no arm branch).

    Args:
        item: The case.
        display_smiles: The molecule as SMILES, shown to every arm (the full ligand
            for edit2d/probe, the scaffold for decorate).

    Returns:
        The ``-p`` text.
    """
    kind = case_type(item)
    if kind == "decorate":
        return (
            "A chemical scaffold is posed in a protein binding pocket. "
            f"{item['instruction']}\n"
            f"The scaffold as SMILES: {display_smiles}\n{_SMILES_CONTRACT}"
        )
    if kind == "probe":
        return (
            f"A ligand is posed in a protein binding pocket. {item['question']}\n"
            f"The ligand as SMILES: {display_smiles}\n{_ANSWER_CONTRACT}"
        )
    return (
        f"{item['instruction']}\n"
        f"The molecule as SMILES: {display_smiles}\n{_SMILES_CONTRACT}"
    )


def mechanics(
    arm: arms.Arm,
    item: dict,
    bind_target: str,
    receptor: str | None,
    pose_out: str | None,
) -> str:
    """The per-arm access mechanics (system-prompt layer a), or "" for naked.

    Says how the arm loads the molecule and its scorer and reports the answer —
    the file paths live here, not in the identical task prompt.

    Args:
        arm: The arm being run.
        item: The case.
        bind_target: What the chemistree arm binds (a SMILES for edit2d, else an SDF
            path); also the SDF path the generalist works from in 3D.
        receptor: Receptor PDB path (3D cases), or None.
        pose_out: Where the chemistree arm writes its pose (decorate), or None.

    Returns:
        The mechanics text, or "" (naked works from the task-prompt SMILES).
    """
    kind = case_type(item)
    if arm.uses_chemistree:
        if kind == "edit2d":
            return (
                f"Edit the molecule with the chemistree tools. Load it: call bind with "
                f'molecule "{bind_target}". When done, call the smiles tool and report '
                "its output as the answer."
            )
        if kind == "decorate":
            return (
                "Edit the molecule with the chemistree tools. Load the posed scaffold: "
                f'call bind with molecule "{bind_target}" and receptor "{receptor}". '
                f'When finished, call write_pose with path "{pose_out}", then call the '
                "smiles tool and report its output as the answer."
            )
        return (
            "Inspect the molecule with the chemistree tools. Load the posed ligand: "
            f'call bind with molecule "{bind_target}" and receptor "{receptor}". Use '
            "the tools to determine the answer."
        )
    if arm.allowed_tools == "Bash":  # generalist
        if kind == "edit2d":
            return "You have a Bash tool with rdkit (python) available."
        smina = os.environ.get("SMINA_BIN", "smina")
        noun = "scaffold" if kind == "decorate" else "ligand"
        text = (
            f"You have a Bash tool with rdkit (python) and smina. The posed {noun} SDF "
            f"is {bind_target} and the receptor is {receptor}."
        )
        if kind == "decorate":
            text += (
                f"\nScore a candidate with:\n  {smina} --receptor {receptor} --ligand "
                f"<candidate.sdf> --autobox_ligand {bind_target} --scoring vinardo "
                "--score_only"
            )
        return text
    return ""  # naked


def guidance_text(arm: arms.Arm) -> str:
    """The medchem playbook plus each arm's tool-usage translation."""
    if arm.uses_chemistree:
        return str(guidance.chemistree_guidance())
    if arm.allowed_tools == "Bash":  # generalist
        return f"{guidance.medchem_guidance()}\n\n{_GENERALIST_GUIDANCE}"
    return str(guidance.medchem_guidance())  # naked


def system_prompt(
    arm: arms.Arm,
    item: dict,
    bind_target: str,
    receptor: str | None,
    pose_out: str | None,
    guidance_on: bool,
) -> str | None:
    """The `--append-system-prompt` text: mechanics + guidance, or None.

    Guidance (the medchem playbook plus the arm's tool-usage translation) is applied
    on every track, matching what the app injects; ``--no-guidance`` drops it (the
    base-prompt ablation).

    Args:
        arm: The arm being run.
        item: The case.
        bind_target: See :func:`mechanics`.
        receptor: Receptor PDB path, or None.
        pose_out: Chemistree pose path (decorate), or None.
        guidance_on: When False (the ablation), the guidance layer is dropped.

    Returns:
        The system-prompt text, or None when nothing applies.
    """
    parts = []
    mech = mechanics(arm, item, bind_target, receptor, pose_out)
    if mech:
        parts.append(mech)
    if guidance_on:
        parts.append(guidance_text(arm))
    return "\n\n".join(parts) if parts else None


def build_command(
    arm: arms.Arm,
    prompt: str,
    model: str,
    mcp_config: Path | None,
    sys_prompt: str | None = None,
) -> list[str]:
    """The ``claude`` argv for one case.

    Args:
        arm: The arm being run.
        prompt: The (identical) task prompt.
        model: The Claude model alias.
        mcp_config: Path to the arm's ``--mcp-config`` file, or None.
        sys_prompt: Guidance to append via ``--append-system-prompt``, or None.

    Returns:
        The argument list to spawn.
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model]
    if sys_prompt:
        cmd += ["--append-system-prompt", sys_prompt]
    if mcp_config is not None:
        cmd += ["--mcp-config", str(mcp_config), "--strict-mcp-config"]
    if arm.allowed_tools:
        cmd += ["--allowedTools", arm.allowed_tools]
    if arm.disallowed_tools:
        cmd += ["--disallowedTools", arm.disallowed_tools]
    return cmd


# --- case preparation and execution --------------------------------------------------


def prepare_case(item: dict, workdir: Path) -> tuple[str, Path | None, str | None]:
    """Compute the shown SMILES and the harness-side input paths (arm-independent).

    Args:
        item: The case.
        workdir: Directory for the stripped scaffold SDF (decorate only).

    Returns:
        ``(display_smiles, seed_sdf, receptor_src)`` — the SMILES shown to every arm,
        the stripped scaffold SDF the harness scores against (decorate only), and the
        source receptor path (3D only). These stay in the repo/workdir for scoring;
        the agent only ever sees the staged copies (see :func:`_stage`).
    """
    kind = case_type(item)
    if kind == "decorate":
        seed_sdf = workdir / f"seed_{item['id']}.sdf"
        display = strip_murcko.write_scaffold(item["target"], str(seed_sdf))
        return display, seed_sdf, item["receptor"]
    if kind == "probe":
        return ligand_smiles(item), None, item["receptor"]
    return item["smiles"], None, None


def _stage(
    arm: arms.Arm,
    item: dict,
    seed_sdf: Path | None,
    receptor_src: str | None,
    stage: Path,
) -> tuple[str, str | None, str | None, Path | None]:
    """Copy the agent-visible inputs into the isolated run dir; return the arm's paths.

    Only the molecule and receptor the arm may see are copied in (generic names). The
    agent runs with ``cwd`` set to ``stage`` (see :func:`run_case`), so its shell can
    reach neither the crystal ligand (the answer) nor the case files (the answer key).
    The chemistree arm gets absolute staged paths (it has no shell to roam with); the
    generalist gets relative names, so no repo path is ever leaked to it.

    Args:
        arm: The arm being run.
        item: The case.
        seed_sdf: The scaffold SDF (decorate) to stage as ``scaffold.sdf``, else None.
        receptor_src: The source receptor path (3D), else None.
        stage: The isolated run directory to copy inputs into.

    Returns:
        ``(bind_target, receptor_arg, agent_pose, pose_file)`` — what the agent loads,
        the receptor path it is given, the path chemistree writes its pose to, and the
        actual pose file the harness reads back.
    """
    kind = case_type(item)
    if kind == "edit2d":
        return item["smiles"], None, None, None
    source = seed_sdf if kind == "decorate" else Path(item["ligand"])
    name = "scaffold.sdf" if kind == "decorate" else "ligand.sdf"
    shutil.copy(str(source), stage / name)
    assert receptor_src is not None
    shutil.copy(receptor_src, stage / "receptor.pdb")
    chem = arm.uses_chemistree
    bind_target = str((stage / name).resolve()) if chem else name
    receptor_arg = str((stage / "receptor.pdb").resolve()) if chem else "receptor.pdb"
    pose_file = stage / "pose.sdf" if (chem and kind == "decorate") else None
    agent_pose = str(pose_file.resolve()) if pose_file else None
    return bind_target, receptor_arg, agent_pose, pose_file


def run_case(
    arm: arms.Arm,
    item: dict,
    model: str,
    timeout: int,
    workdir: Path,
    trace: bool = False,
    guidance_on: bool = True,
    replicate: int = 1,
) -> dict:
    """Run one case and return its result row (prompts, metrics, and scoring).

    Args:
        arm: The arm to run.
        item: The case.
        model: The Claude model alias.
        timeout: Seconds before the case is abandoned.
        workdir: Directory for the mcp-config and scratch files.
        trace: Whether the chemistree server logs a per-tool-call trace.
        guidance_on: When False, the medchem guidance is dropped (the ablation).
        replicate: 1-based sample index; recorded on the row and used to keep each
            sample's trace file distinct when ``--repeat`` runs a case more than once.

    Returns:
        The result row.
    """
    kind = case_type(item)
    display_smiles, seed_sdf, receptor_src = prepare_case(item, workdir)
    # Isolated run dir: the agent's cwd holds only what this arm may see, so its shell
    # cannot reach the crystal ligand (the answer) or the case files (the answer key).
    stage = workdir / f"stage_{item['id']}_{arm.name}"
    stage.mkdir(parents=True, exist_ok=True)
    bind_target, receptor_arg, agent_pose, pose_file = _stage(
        arm, item, seed_sdf, receptor_src, stage
    )

    mcp_config = None
    trace_path = None
    if arm.uses_chemistree:
        if trace:
            trace_path = workdir / f"trace_{item['id']}_r{replicate}.jsonl"
        mcp_config = workdir / f"mcp_{item['id']}.json"
        config = arms.chemistree_mcp_config(
            arm.tools_profile, str(trace_path) if trace_path else None
        )
        mcp_config.write_text(json.dumps(config))

    task_prompt = build_prompt(item, display_smiles)
    sys_prompt = system_prompt(
        arm, item, bind_target, receptor_arg, agent_pose, guidance_on
    )
    command = build_command(arm, task_prompt, model, mcp_config, sys_prompt)

    row: dict = {
        "id": item["id"],
        "replicate": replicate,
        "arm": arm.name,
        "model": model,
        "guidance": guidance_on,
        "prompt": task_prompt,
        "system_prompt": sys_prompt,
    }
    start = time.time()
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, cwd=str(stage)
        )
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
    reply = result.get("result", "")
    if trace_path is not None and trace_path.exists():
        row["trace"] = str(trace_path)

    if kind == "probe":
        _score_probe(row, item, reply)
    elif kind == "decorate":
        final = metrics.parse_final_smiles(reply)
        row["final_smiles"] = final
        row["status"] = "ok" if final else "no_final_smiles"
        row.update(
            _score_3d(arm, item, final, seed_sdf, display_smiles, pose_file, workdir)
        )
    else:  # edit2d
        final = metrics.parse_final_smiles(reply)
        row["final_smiles"] = final
        row["status"] = "ok" if final else "no_final_smiles"
        row["category"] = item.get("category")
        row["gold"] = item["gold"]
        row["correct"] = canonical(final) is not None and canonical(final) == canonical(
            item["gold"]
        )
    return row


def _score_probe(row: dict, item: dict, reply: str) -> None:
    """Score a B2 understanding probe by matching the answer (mutates ``row``)."""
    answer = metrics.parse_final_answer(reply)
    row["answer"] = answer
    row["expected"] = item["answer"]
    row["status"] = "ok" if answer else "no_final_answer"
    row["correct"] = answer_matches(item["answer"], answer)


def _score_3d(
    arm: arms.Arm,
    item: dict,
    final: str | None,
    seed_sdf: Path | None,
    seed_smiles: str,
    pose_out: Path | None,
    workdir: Path,
) -> dict:
    """Score a decorate case: two binding columns + recovery.

    Column 1 redocks each arm's final SMILES (fair). Column 2 scores the chemistree
    ``write_pose`` output SDF directly with smina ``--score_only`` — the pose it
    actually built, not a re-docked one. Recovery is ECFP4 Tanimoto to the crystal.

    Args:
        arm: The arm being scored.
        item: The decorate case (crystal ``target``, ``receptor``).
        final: The arm's final SMILES, or None.
        seed_sdf: The posed scaffold the agent started from.
        seed_smiles: The scaffold's SMILES.
        pose_out: The chemistree arm's written pose SDF, or None.
        workdir: Directory for smina's scratch files.

    Returns:
        Scoring fields to merge onto the row.
    """
    receptor, box = item["receptor"], item["target"]
    target_smi = target_smiles(item)
    baseline = oracle_smina.redock(seed_smiles, receptor, box, workdir)
    final_aff = oracle_smina.redock(final, receptor, box, workdir) if final else None
    out: dict = {"redock_scaffold": baseline, "redock_final": final_aff}
    if baseline is not None and final_aff is not None:
        out["redock_delta"] = round(final_aff - baseline, 2)  # negative = better
    out["recovery_scaffold"] = tanimoto(seed_smiles, target_smi)  # floor (~0.33)
    out["recovery_final"] = tanimoto(final, target_smi)
    if arm.uses_chemistree and pose_out is not None and pose_out.exists():
        out["pose_final"] = oracle_smina.score_pose(str(pose_out), receptor)
        out["pose_scaffold"] = oracle_smina.score_pose(str(seed_sdf), receptor)
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
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Samples per case (haiku is stochastic). Each case is run this many "
        "times, writing one row per sample to --out, tagged with a 'replicate' index.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-case seconds: a generous hang-guard; efficiency is cost/tokens.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the identical task prompt and every arm's system prompt, without "
        "running (the parity check).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Log the chemistree per-tool-call trace (extra disk writes; off by "
        "default).",
    )
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help="Ablation: skip the medchem guidance layer, isolating the representation "
        "from the prompt (decorate only).",
    )
    args = parser.parse_args()

    arm = arms.ARMS[args.arm]
    items = load_items(Path(args.items))
    if args.limit is not None:
        items = items[: args.limit]
    workdir = Path(tempfile.mkdtemp(prefix="chemistree-bench-"))

    if args.dry_run:
        _dry_run(items, workdir, guidance_on=not args.no_guidance)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for item in items:
            for replicate in range(1, args.repeat + 1):
                row = run_case(
                    arm,
                    item,
                    args.model,
                    args.timeout,
                    workdir,
                    args.trace,
                    guidance_on=not args.no_guidance,
                    replicate=replicate,
                )
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                tag = f"{row['id']} r{replicate}" if args.repeat > 1 else row["id"]
                print(f"[{row['status']}] {tag} ({arm.name}) {_summary(row)}")


def _summary(row: dict) -> str:
    """A one-line result summary for the console."""
    if "redock_delta" in row or row.get("recovery_final") is not None:
        head = (
            f"Δaffinity={row.get('redock_delta')} "
            f"recovery={row.get('recovery_scaffold')}->{row.get('recovery_final')}"
        )
    else:
        head = f"correct={row.get('correct')}"
    return (
        f"{head} footprint={row.get('context_footprint')} cost=${row.get('cost_usd')}"
    )


def _dry_run(items: list[dict], workdir: Path, guidance_on: bool) -> None:
    """Print the shared task prompt and each arm's system prompt (the parity check)."""
    for item in items:
        display_smiles, seed_sdf, receptor_src = prepare_case(item, workdir)
        print(f"\n===== {item['id']} ({case_type(item)}) =====")
        print("--- TASK PROMPT (identical across arms) ---")
        print(build_prompt(item, display_smiles))
        for arm in arms.ARMS.values():
            stage = workdir / f"stage_{item['id']}_{arm.name}"
            stage.mkdir(parents=True, exist_ok=True)
            bind_target, receptor_arg, agent_pose, _ = _stage(
                arm, item, seed_sdf, receptor_src, stage
            )
            sp = system_prompt(
                arm, item, bind_target, receptor_arg, agent_pose, guidance_on
            )
            print(f"--- SYSTEM PROMPT [{arm.name}] ---")
            print(sp)


if __name__ == "__main__":
    main()
