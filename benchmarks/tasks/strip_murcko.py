"""Strip a posed ligand to its Murcko scaffold, keeping its crystal coordinates.

The scaffold seeds a recovery case (Track B): the agent re-elaborates it, and the
original ligand is the ground-truth target. Coordinates are preserved so the
scaffold stays posed in the pocket.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_pose(ligand_sdf: str) -> Chem.Mol:
    """The Murcko scaffold of a posed ligand, with the scaffold atoms' coordinates.

    Removes every side-chain atom from a copy of the posed ligand, so the remaining
    scaffold keeps its crystal 3D coordinates; the stripped positions become
    implicit hydrogens.

    Args:
        ligand_sdf: Path to a posed ligand SDF/MOL file.

    Returns:
        The scaffold as a molecule carrying a 3D conformer.
    """
    mol = Chem.RemoveHs(Chem.MolFromMolFile(ligand_sdf, removeHs=False))
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # GetScaffoldForMol returns a valid molecule (RDKit keeps the exocyclic double
    # bonds a ring needs). Give it the scaffold atoms' crystal coordinates by
    # matching it back onto the posed ligand, rather than deleting atoms from the
    # aromatic parent (which would break kekulization).
    match = mol.GetSubstructMatch(scaffold)
    source = mol.GetConformer()
    conf = Chem.Conformer(scaffold.GetNumAtoms())
    for scaffold_idx, mol_idx in enumerate(match):
        conf.SetAtomPosition(scaffold_idx, source.GetAtomPosition(mol_idx))
    scaffold.RemoveAllConformers()
    scaffold.AddConformer(conf, assignId=True)
    return scaffold


def write_scaffold(ligand_sdf: str, out_sdf: str) -> str:
    """Write the posed Murcko scaffold to an SDF and return its canonical SMILES.

    Args:
        ligand_sdf: Path to the posed ligand.
        out_sdf: Path to write the scaffold SDF to.

    Returns:
        The scaffold's canonical SMILES.
    """
    scaf = scaffold_pose(ligand_sdf)
    Chem.MolToMolFile(scaf, out_sdf)
    return str(Chem.MolToSmiles(scaf))
