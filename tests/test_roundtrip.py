"""Round-trip fidelity: fragmenting then reconstructing recovers the input."""

import pytest
from rdkit import Chem

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
