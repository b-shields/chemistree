"""Fragment naming and coarse classification."""

import pytest
from rdkit import Chem

from chemistree import classify_fragment, name_fragment
from chemistree.fragment import Fragment
from chemistree.naming import group_smiles


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


@pytest.mark.parametrize(
    "smiles,name",
    [
        ("[*]c1ccncc1", "pyridine"),
        ("[1*]c1ccc([2*])cc1", "benzene"),  # multi-port ring, no exact match
        ("[*]N1CCOCC1", "morpholine"),
        ("[*]c1ccc2[nH]ccc2c1", "indole"),  # fused: not mis-named "benzene"
    ],
)
def test_ring_system_names(smiles, name):
    assert name_fragment(_fragment(smiles)) == name


def test_phenyl_beats_benzene_for_single_port():
    # A one-port benzene is the phenyl substituent (exact match wins).
    assert name_fragment(_fragment("[*]c1ccccc1")) == "phenyl"


def test_unknown_fragment_has_no_name():
    assert name_fragment(_fragment("[*]CCCCCCCC")) is None  # octyl: no name, no ring


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


def test_group_smiles_resolves_bounded_synonyms():
    # A few common alt/adjective forms alias the canonical curated names.
    assert group_smiles("hydroxy") == group_smiles("hydroxyl") == "*O"
    assert group_smiles("nitrile") == group_smiles("cyano") == "*C#N"


def test_group_smiles_unknown_name_is_none():
    assert group_smiles("flibberto") is None
