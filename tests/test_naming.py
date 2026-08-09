"""Fragment naming and coarse classification."""

import pytest
from rdkit import Chem

from chemistree import classify_fragment, name_fragment
from chemistree.fragment import Fragment


def _fragment(smiles: str) -> Fragment:
    return Fragment(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    "smiles,name",
    [
        ("[*]C", "methyl"),
        ("[*]CC", "ethyl"),
        ("[*]C(F)(F)F", "trifluoromethyl"),
        ("[*]c1ccccc1", "phenyl"),
        ("[*]O", "hydroxyl"),
    ],
)
def test_curated_names(smiles, name):
    assert name_fragment(_fragment(smiles)) == name


def test_name_is_label_independent():
    # The pairing isotope on the dummy must not affect the lookup.
    assert name_fragment(_fragment("[7*]C")) == "methyl"


def test_unknown_fragment_has_no_name():
    assert name_fragment(_fragment("[*]c1ccncc1")) is None


@pytest.mark.parametrize(
    "smiles,classification",
    [
        ("[*]C", "alkyl"),
        ("[*]CCC", "alkyl"),
        ("[*]c1ccccc1", "aromatic"),
        ("[*]c1ccncc1", "heteroaromatic"),
        ("[*]O", "other"),
        ("[*]C(=O)O", "other"),
    ],
)
def test_classification(smiles, classification):
    assert classify_fragment(_fragment(smiles)) == classification
