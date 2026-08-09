"""Functional groups: kept intact during fragmentation, named as units."""

from rdkit import Chem

from chemistree import annotate, fragment, prepare_molecule
from chemistree.functional_groups import functional_group_name, protects_bond


def _node_names(smiles: str) -> set[str | None]:
    tree = fragment(prepare_molecule(smiles, three_d=False))
    return {node.name for node in annotate(tree).nodes}


def test_ester_and_acid_kept_whole_in_aspirin():
    # Previously the ester atomized into carbonyl + ether; now it is one node.
    assert {"benzene", "ester", "carboxyl", "methyl"} <= _node_names(
        "CC(=O)Oc1ccccc1C(=O)O"
    )


def test_amide_kept_whole():
    assert "amide" in _node_names("CC(=O)Nc1ccccc1")


def test_ester_internal_bond_is_protected():
    mol = Chem.MolFromSmiles("CC(=O)OC")  # methyl acetate
    match = mol.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=[OX1])[OX2]"))
    bond = mol.GetBondBetweenAtoms(match[0], match[2])  # the carbonyl C - ester O
    assert protects_bond(mol, bond) is True


def test_functional_group_name_prefers_most_specific():
    # Methyl carbamate contains ester and amide motifs; carbamate must win.
    assert functional_group_name(Chem.MolFromSmiles("COC(=O)N")) == "carbamate"
    assert functional_group_name(Chem.MolFromSmiles("CC(=O)OCC")) == "ester"
    assert functional_group_name(Chem.MolFromSmiles("CCCC")) is None
