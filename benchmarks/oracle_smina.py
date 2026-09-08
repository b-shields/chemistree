"""smina oracle: redock a molecule and score a pose, both with Vinardo.

smina lives in its own conda env (its openbabel dependency conflicts with the
chemistree env's Python 3.14 pin). Point ``SMINA_BIN`` at the binary, or have
``smina`` on ``PATH``. Redocking gives the fair, arm-agnostic score from a SMILES;
``--score_only`` scores a pose an arm already produced.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

SMINA_BIN = os.environ.get("SMINA_BIN", "smina")
_SEED = 0
_EXHAUSTIVENESS = 8
_BOX_PAD = 4  # angstrom padding around the crystal ligand for the autobox
# The best docked mode is row 1 of smina's table; --score_only prints "Affinity:".
_DOCK_AFFINITY = re.compile(r"^\s*1\s+(-?\d+\.\d+)", re.MULTILINE)
_SCORE_AFFINITY = re.compile(r"Affinity:\s*(-?\d+\.\d+)")


def _embed(smiles: str, path: Path) -> bool:
    """Write a 3D conformer of a SMILES to an SDF file.

    Args:
        smiles: The molecule to embed.
        path: SDF path to write.

    Returns:
        True on success; False if the SMILES is unparseable or embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=_SEED) != 0:
        return False
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, str(path))
    return True


def redock(
    smiles: str, receptor: str, autobox_ligand: str, workdir: Path
) -> float | None:
    """Dock a SMILES into the receptor box and return the best Vinardo affinity.

    The box is taken from the crystal ligand (``--autobox_ligand``). A fresh 3D
    conformer is embedded, then smina searches poses.

    Args:
        smiles: The molecule to dock.
        receptor: Receptor file path (PDB or PDBQT).
        autobox_ligand: Crystal-ligand file defining the box.
        workdir: Directory for the embedded ligand and docked output.

    Returns:
        The best pose's affinity (lower is better), or None if embedding or
        docking failed.
    """
    lig = workdir / "dock_in.sdf"
    if not _embed(smiles, lig):
        return None
    out = workdir / "dock_out.sdf"
    result = subprocess.run(
        [
            SMINA_BIN,
            "--receptor",
            receptor,
            "--ligand",
            str(lig),
            "--autobox_ligand",
            autobox_ligand,
            "--autobox_add",
            str(_BOX_PAD),
            "--scoring",
            "vinardo",
            "--seed",
            str(_SEED),
            "--exhaustiveness",
            str(_EXHAUSTIVENESS),
            "--cpu",
            "1",
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    match = _DOCK_AFFINITY.search(result.stdout)
    return float(match.group(1)) if match else None


def score_pose(pose_sdf: str, receptor: str) -> float | None:
    """Score a given pose with Vinardo, without docking (``--score_only``).

    Args:
        pose_sdf: SDF file of the posed ligand, in the receptor frame.
        receptor: Receptor file path.

    Returns:
        The pose's affinity (lower is better), or None if smina reported none.
    """
    result = subprocess.run(
        [
            SMINA_BIN,
            "--receptor",
            receptor,
            "--ligand",
            pose_sdf,
            "--scoring",
            "vinardo",
            "--score_only",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    match = _SCORE_AFFINITY.search(result.stdout)
    return float(match.group(1)) if match else None
