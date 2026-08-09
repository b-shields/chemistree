"""Round-trip fidelity: fragmenting then reconstructing recovers the input."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from chemistree import fragment

ROUND_TRIP_SMILES = [
    "Cc1ccccc1",  # toluene
    "CCc1ccccc1",  # ethylbenzene
    "c1ccc(-c2ccccc2)cc1",  # biphenyl
    "CC(=O)c1ccccc1",  # acetophenone
    "Cc1ccc(Cl)cc1",  # 4-chlorotoluene
    "CCCCO",  # butanol
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "CCC",  # propane (no breakable bonds)
]


def _canonical(smiles: str) -> str:
    return str(Chem.CanonSmiles(smiles))


@pytest.mark.parametrize("smiles", ROUND_TRIP_SMILES)
def test_reconstruct_recovers_input(smiles):
    tree = fragment(Chem.MolFromSmiles(smiles))
    rebuilt = tree.reconstruct()
    assert Chem.MolToSmiles(rebuilt) == _canonical(smiles)


def _coordinate_set(mol: Chem.Mol) -> list[tuple]:
    """Sorted (element, x, y, z) tuples; reorder-safe for comparing conformers."""
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append(
            (atom.GetAtomicNum(), round(p.x, 4), round(p.y, 4), round(p.z, 4))
        )
    return sorted(coords)


@pytest.mark.parametrize("smiles", ["Cc1ccccc1", "COC(=O)c1ccc(C)cc1", "CCCCO"])
def test_reconstruct_preserves_coordinates(smiles):
    # Embed a 3D conformer, then confirm fragmentation leaves every atom in place.
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol = Chem.RemoveHs(mol)

    rebuilt = fragment(mol).reconstruct()

    assert rebuilt.GetNumConformers() == 1
    assert _coordinate_set(rebuilt) == _coordinate_set(mol)
