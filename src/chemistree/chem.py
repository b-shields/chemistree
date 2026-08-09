"""Molecule preparation.

The 3D edit path assumes explicit hydrogens so that coordinates and
stereochemistry are unambiguous. ``prepare_molecule`` normalizes inputs to that
convention (or to a connectivity-only form for the 2D path).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

_EMBED_SEED = 0xF00D  # fixed so embedding is reproducible


def prepare_molecule(
    mol: str | Chem.Mol, *, three_d: bool = True, seed: int = _EMBED_SEED
) -> Chem.Mol:
    """Normalize a molecule for editing.

    In the 3D path the result has explicit hydrogens and a 3D conformer: an
    existing 3D pose is kept (hydrogens added in place, preserving coordinates),
    otherwise a conformer is embedded. In the 2D path the result is connectivity
    only, with implicit hydrogens.

    Args:
        mol: A SMILES string or RDKit Mol.
        three_d: Whether to guarantee explicit hydrogens and a 3D conformer.
        seed: Embedding seed, for reproducible coordinates.

    Returns:
        The prepared molecule.

    Raises:
        ValueError: If the input cannot be parsed.
    """
    mol = _as_mol(mol)
    if not three_d:
        return Chem.RemoveHs(mol)
    if mol.GetNumConformers() and mol.GetConformer().Is3D():
        return Chem.AddHs(mol, addCoords=True)  # keep the pose, add explicit Hs
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _as_mol(mol: str | Chem.Mol) -> Chem.Mol:
    """Return a fresh, sanitized molecule from a SMILES string or Mol.

    Raises:
        ValueError: If a SMILES string cannot be parsed.
    """
    if isinstance(mol, str):
        parsed = Chem.MolFromSmiles(mol)
    else:
        parsed = Chem.Mol(mol)
        Chem.SanitizeMol(parsed)
    if parsed is None:
        raise ValueError(f"could not parse molecule: {mol!r}")
    return parsed
