"""Fetch and prep DUD-Z targets: a receptor PDB and a posed reference-ligand SDF.

Both benchmark tracks need, per target, just two files — the receptor and the crystal
ligand posed in its pocket (the same pair ``tests/data/abl1/`` holds by hand). DUD-Z
serves each target's ``rec.crg.pdb`` (receptor) and ``xtal-lig.pdb`` (crystal ligand),
but the ligand PDB carries no bond orders. This module rebuilds correct bond orders by
transferring them from the ligand's RCSB *ideal* SDF onto the crystal coordinates.

There are no tests for this (as for the rest of ``benchmarks/``). Instead a three-guard
identity gate runs inside the build and **refuses** to emit a reference it cannot prove
is the real crystal ligand — a wrong RCSB match or a truncated/ambiguous pose raises,
and the target is skipped and logged rather than silently mis-prepped:

1. one unambiguous ligand residue (same three-letter code);
2. its heavy-atom element composition equals the RCSB template's;
3. the bond-order transfer succeeds and reproduces the template's constitution.

The reference then takes its stereochemistry from the crystal 3D coordinates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from rdkit import Chem
from rdkit.Chem import AllChem

DUDZ_BASE = "https://dudez.docking.org/DOCKING_GRIDS_AND_POSES"
RCSB_IDEAL = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"

# Ten DUD-Z targets spanning ten target classes, each with a drug-like crystal ligand,
# all verified to pass the identity gate. abl1 is deliberately absent — it is the
# dev/test case, already prepared by hand in tests/data/abl1.
DEFAULT_TARGETS: tuple[str, ...] = (
    "EGFR",  # receptor tyrosine kinase
    "AA2AR",  # GPCR (adenosine A2A)
    "ANDR",  # nuclear receptor (androgen)
    "HIVPR",  # aspartic protease
    "FA10",  # serine protease (factor Xa)
    "HDAC8",  # metalloenzyme (deacetylase)
    "PARP1",  # transferase (PARP)
    "HS90A",  # chaperone ATPase
    "ADA",  # deaminase
    "NRAM",  # glycoside hydrolase (neuraminidase)
)


class ResidueKey(NamedTuple):
    """A ligand residue's identity: chain, sequence number, insertion code, name."""

    chain: str
    resseq: str
    icode: str
    resname: str


def _download(url: str, dest: Path) -> None:
    """Download a URL to a file, raising on any HTTP error.

    Args:
        url: The URL to fetch.
        dest: File path to write the bytes to.

    Raises:
        urllib.error.HTTPError: If the server returns an error status (e.g. a
            missing ligand code gives 404).
    """
    with urllib.request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


def ligand_residues(xtal_pdb: Path) -> list[tuple[ResidueKey, str]]:
    """Group a ligand PDB's atom records by residue, in file order.

    Args:
        xtal_pdb: Path to a crystal-ligand PDB (ATOM/HETATM records only).

    Returns:
        ``(residue_key, pdb_block)`` pairs — one per distinct residue, each block the
        residue's ATOM/HETATM lines joined as they appear.
    """
    blocks: dict[ResidueKey, list[str]] = {}
    for line in xtal_pdb.read_text().splitlines():
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        key = ResidueKey(line[21], line[22:26].strip(), line[26], line[17:20].strip())
        blocks.setdefault(key, []).append(line)
    return [(key, "\n".join(lines)) for key, lines in blocks.items()]


def ligand_code(residues: list[tuple[ResidueKey, str]]) -> str:
    """The single three-letter ligand code shared by every residue.

    Args:
        residues: The output of :func:`ligand_residues`.

    Returns:
        The residue name to look the ligand up by at RCSB.

    Raises:
        ValueError: If the residues carry more than one distinct name (ambiguous —
            e.g. a co-crystallized additive), so no single code is trustworthy.
    """
    names = {key.resname for key, _ in residues}
    if len(names) != 1:
        raise ValueError(f"ambiguous ligand residues: {sorted(names)}")
    return names.pop()


def _heavy_composition(mol: Chem.Mol) -> Counter:
    """The molecule's heavy-atom element counts (hydrogens excluded)."""
    return Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _pose_from_block(block: str) -> Chem.Mol:
    """Parse a ligand PDB block to a molecule, keeping coordinates, no sanitizing."""
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("unparseable ligand PDB block")
    return Chem.RemoveHs(mol, sanitize=False)


def select_pose(residues: list[tuple[ResidueKey, str]], template: Chem.Mol) -> Chem.Mol:
    """The crystal-ligand pose whose composition matches the template (guards 1-2).

    Picks the first residue whose heavy-atom element composition equals the template's.
    This rejects a wrong RCSB match (different formula) and, given more than one
    residue, selects the ligand copy over any additive.

    Args:
        residues: The output of :func:`ligand_residues`.
        template: The RCSB ideal molecule (bond-order source).

    Returns:
        The matching pose as an unsanitized molecule carrying the crystal coordinates.

    Raises:
        ValueError: If no residue matches the template composition.
    """
    want = _heavy_composition(template)
    for _, block in residues:
        pose = _pose_from_block(block)
        if _heavy_composition(pose) == want:
            return pose
    raise ValueError(f"no ligand residue matches the template composition {dict(want)}")


def build_reference(xtal_pdb: Path, ideal_sdf: Path, out_sdf: Path) -> str:
    """Build a posed reference SDF with correct bond orders, through the identity gate.

    Transfers bond orders from the RCSB ideal template onto the crystal coordinates,
    verifies the result is constitutionally the template molecule (guard 3, stereo
    ignored), assigns stereochemistry from the 3D pose, and writes the SDF.

    Args:
        xtal_pdb: The DUD-Z crystal-ligand PDB (posed, no bond orders).
        ideal_sdf: The ligand's RCSB ideal SDF (bond-order template).
        out_sdf: Path to write the posed reference SDF to.

    Returns:
        The reference's canonical (isomeric) SMILES.

    Raises:
        ValueError: If any identity guard fails — the target must then be skipped.
    """
    template = Chem.MolFromMolFile(str(ideal_sdf))
    if template is None:
        raise ValueError("unparseable RCSB ideal SDF")
    template = Chem.RemoveHs(template)
    pose = select_pose(ligand_residues(xtal_pdb), template)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, pose)
    except ValueError as exc:
        raise ValueError(f"bond-order transfer failed: {exc}") from exc

    def constitution(mol: Chem.Mol) -> str:
        return str(Chem.MolToSmiles(mol, isomericSmiles=False))

    if constitution(fixed) != constitution(template):
        raise ValueError("built graph does not match the template constitution")
    Chem.AssignStereochemistryFrom3D(fixed)
    Chem.MolToMolFile(fixed, str(out_sdf))
    return str(Chem.MolToSmiles(fixed))


def fetch_target(target: str, out_dir: Path) -> dict:
    """Fetch and prep one DUD-Z target into ``out_dir/{target}/``.

    Downloads the receptor and crystal ligand, resolves the ligand code, pulls the RCSB
    ideal template, and builds the posed reference through the identity gate. The
    receptor is kept as-is (bare prep): every arm and the baseline share this one
    receptor, so systematic prep effects cancel in the comparison.

    Args:
        target: The DUD-Z target name (e.g. ``"EGFR"``).
        out_dir: Base output directory; files land under ``out_dir/{target_lower}/``.

    Returns:
        A manifest row: target, ligand_code, smiles, receptor, reference, status "ok".

    Raises:
        ValueError: If the identity gate rejects the ligand (caller skips the target).
        urllib.error.HTTPError: If a download fails.
    """
    dest = out_dir / target.lower()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        xtal = tmp_path / "xtal-lig.pdb"
        _download(f"{DUDZ_BASE}/{target}/xtal-lig.pdb", xtal)
        code = ligand_code(ligand_residues(xtal))
        ideal = tmp_path / "ideal.sdf"
        _download(RCSB_IDEAL.format(code=code), ideal)

        receptor = dest / "receptor.pdb"
        _download(f"{DUDZ_BASE}/{target}/rec.crg.pdb", tmp_path / "rec.pdb")
        shutil.copy(tmp_path / "rec.pdb", receptor)

        reference = dest / "reference.sdf"
        smiles = build_reference(xtal, ideal, reference)
    return {
        "target": target,
        "ligand_code": code,
        "smiles": smiles,
        "receptor": str(receptor),
        "reference": str(reference),
        "status": "ok",
    }


def main() -> None:
    """Fetch the target set, writing per-target files and a provenance manifest."""
    parser = argparse.ArgumentParser(description="Fetch and prep DUD-Z targets.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="DUD-Z target names (default: the ten-target benchmark set).",
    )
    parser.add_argument(
        "--out", default="benchmarks/data", help="Base output directory."
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w") as handle:
        for target in args.targets:
            try:
                row = fetch_target(target, out_dir)
            except Exception as exc:  # skip-and-log: never emit an unproven reference
                row = {"target": target, "status": "skip", "reason": str(exc)}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if row["status"] == "ok":
                print(f"[ok]   {target}: {row['ligand_code']}  {row['smiles']}")
            else:
                print(f"[skip] {target}: {row['reason']}")


if __name__ == "__main__":
    main()
