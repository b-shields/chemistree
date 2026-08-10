"""Positional selection: topological distance and the ring synonym shim."""

import pytest
from rdkit import Chem

from chemistree.selection import atoms_at_distance, is_open_position, resolve_offset


def test_synonyms_map_to_distances():
    assert resolve_offset("ortho") == 1
    assert resolve_offset("meta") == 2
    assert resolve_offset("para", ring_size=6) == 3


def test_integer_offset_passes_through():
    assert resolve_offset(4) == 4


def test_para_requires_six_membered_ring():
    with pytest.raises(ValueError, match="para"):
        resolve_offset("para", ring_size=5)
    with pytest.raises(ValueError, match="para"):
        resolve_offset("para")  # ring_size unknown


def test_unknown_term_raises():
    with pytest.raises(ValueError, match="unknown position"):
        resolve_offset("distal")


def test_ortho_meta_para_on_benzene():
    benzene = Chem.MolFromSmiles("c1ccccc1")
    assert len(atoms_at_distance(benzene, 0, resolve_offset("ortho"))) == 2
    assert len(atoms_at_distance(benzene, 0, resolve_offset("meta"))) == 2
    assert atoms_at_distance(benzene, 0, resolve_offset("para", ring_size=6)) == [3]


def test_distance_counts_on_a_chain():
    pentane = Chem.MolFromSmiles("CCCCC")
    assert atoms_at_distance(pentane, 0, 1) == [1]
    assert atoms_at_distance(pentane, 0, 2) == [2]
    assert atoms_at_distance(pentane, 0, 4) == [4]


def test_open_position_excludes_substituted_and_port_atoms():
    phenyl = Chem.MolFromSmiles("[*]c1ccccc1")  # atom 1 bears the port
    anchor = phenyl.GetAtomWithIdx(1)
    assert is_open_position(phenyl, 1) is False  # carries the port, no H
    free = next(n.GetIdx() for n in anchor.GetNeighbors() if n.GetAtomicNum() == 6)
    assert is_open_position(phenyl, free) is True
