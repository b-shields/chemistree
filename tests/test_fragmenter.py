"""Fragmentation behavior: which bonds break and the resulting tree shape."""

from rdkit import Chem

from chemistree import fragment
from chemistree.fragmenter import should_break


def _breakable_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return sum(should_break(mol, b) for b in mol.GetBonds())


def test_toluene_breaks_ring_methyl_bond():
    assert _breakable_count("Cc1ccccc1") == 1


def test_ethyl_stays_whole():
    # Only the ring-CH2 bond breaks; the CH2-CH3 bond inside ethyl does not.
    assert _breakable_count("CCc1ccccc1") == 1


def test_biphenyl_breaks_inter_ring_bond():
    assert _breakable_count("c1ccc(-c2ccccc2)cc1") == 1


def test_alkane_has_no_breakable_bonds():
    assert _breakable_count("CCC") == 0


def test_ring_bonds_never_break():
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert not any(should_break(mol, b) for b in mol.GetBonds())


def test_carbonyl_is_a_functional_group_boundary():
    # Acetophenone: ring-C(=O) (ring rule) and C(=O)-CH3 (FG boundary) both break.
    assert _breakable_count("CC(=O)c1ccccc1") == 2


def test_explicit_hydrogens_are_not_broken():
    # With explicit Hs, ring C-H bonds have a ring-atom endpoint but must not break.
    mol = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    tree = fragment(mol)
    assert len(tree.nodes) == 2  # methyl + ring; every H stays attached
    rebuilt = Chem.RemoveHs(tree.reconstruct())
    assert Chem.MolToSmiles(rebuilt) == Chem.CanonSmiles("Cc1ccccc1")


def test_toluene_fragments_into_two_nodes():
    tree = fragment(Chem.MolFromSmiles("Cc1ccccc1"))
    assert len(tree.nodes) == 2
    smiles = sorted(node.current.smiles for node in tree.nodes)
    assert smiles == sorted(["[1*]C", "[1*]c1ccccc1"])


def test_leaves_excludes_internal_node():
    # 1,2,4-trisubstituted benzene: ring is internal (degree 3), 3 substituent leaves.
    tree = fragment(Chem.MolFromSmiles("Cc1ccc(C)c(C)c1"))
    assert len(tree.leaves()) == 3
