"""Tests for the physicochemical profile and structure-alert screening.

Ground-truth property values are the standard reference figures for small
molecules (benzene, aspirin); alert names are RDKit catalog entry descriptions.
"""

import pytest
from rdkit import Chem

from chemistree.properties import (
    compute_properties,
    profile_markdown,
    structure_alerts,
)


def _mol(smiles: str) -> Chem.Mol:
    """Parse a SMILES to a molecule for a test."""
    return Chem.MolFromSmiles(smiles)


def test_benzene_profile_matches_the_known_values():
    props = compute_properties(_mol("c1ccccc1"))
    assert props.molecular_weight == pytest.approx(78.11, abs=0.01)
    assert props.clogp == pytest.approx(1.69, abs=0.01)
    assert props.tpsa == pytest.approx(0.0, abs=0.01)
    assert props.h_bond_donors == 0
    assert props.h_bond_acceptors == 0
    assert props.rotatable_bonds == 0
    assert props.aromatic_rings == 1
    assert props.fraction_csp3 == pytest.approx(0.0)
    assert props.formal_charge == 0


def test_aspirin_profile_matches_reference_values():
    props = compute_properties(_mol("CC(=O)Oc1ccccc1C(=O)O"))
    assert props.molecular_weight == pytest.approx(180.16, abs=0.01)
    assert props.tpsa == pytest.approx(63.6, abs=0.1)
    assert props.h_bond_donors == 1
    assert props.h_bond_acceptors == 3
    assert props.rotatable_bonds == 2


def test_formal_charge_counts_a_cationic_amine():
    props = compute_properties(_mol("c1ccccc1N2CC[NH2+]CC2"))
    assert props.formal_charge == 1


def test_explicit_hydrogens_do_not_change_the_profile():
    flat = compute_properties(_mol("c1ccccc1"))
    with_h = compute_properties(Chem.AddHs(_mol("c1ccccc1")))
    assert with_h == flat


def test_a_clean_molecule_triggers_no_alerts():
    assert structure_alerts(_mol("c1ccccc1")) == []


def test_a_nitro_group_triggers_a_brenk_alert():
    assert "nitro_group" in structure_alerts(_mol("c1ccccc1[N+](=O)[O-]"))


def test_profile_markdown_lists_properties_then_alerts():
    text = profile_markdown(_mol("c1ccccc1[N+](=O)[O-]"))  # nitrobenzene, MW 123
    assert "**Properties:**" in text
    assert "MW 123" in text
    assert "**Structure alerts:** nitro_group" in text


def test_profile_markdown_says_none_when_clean():
    assert "**Structure alerts:** none" in profile_markdown(_mol("c1ccccc1"))
