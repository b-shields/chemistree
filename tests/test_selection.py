"""Positional selection: topological distance and the ring synonym shim."""

import pytest
from rdkit import Chem

from chemistree import fragment, prepare_molecule
from chemistree.selection import (
    Ambiguous,
    NotFound,
    atoms_at_distance,
    attachment_atom,
    is_free_aromatic_carbon,
    is_open_position,
    resolve_offset,
    resolve_position,
    resolve_site,
    select,
    select_one,
)


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


def test_select_by_name():
    tree = fragment(prepare_molecule("Cc1ccccc1", three_d=False))
    (methyl,) = select(tree, name="methyl")
    assert methyl.current.smiles == "[1*]C"


def test_select_one_not_found():
    tree = fragment(prepare_molecule("Cc1ccccc1", three_d=False))
    with pytest.raises(NotFound):
        select_one(tree, description="ethyl", name="ethyl")


def test_select_one_ambiguous_reports_candidates():
    tree = fragment(
        prepare_molecule("Cc1ccccc1C", three_d=False)
    )  # o-xylene: 2 methyls
    with pytest.raises(Ambiguous) as info:
        select_one(tree, description="methyl", name="methyl")
    assert len(info.value.candidates) == 2


def test_attachment_atom_is_the_ring_carbon_bearing_the_substituent():
    tree = fragment(prepare_molecule("Cc1ccccc1", three_d=False))
    ring = select_one(tree, description="phenyl", name="phenyl")
    methyl = select_one(tree, description="methyl", name="methyl", neighbor_of=ring)

    anchor = attachment_atom(tree, ring, methyl)
    atom = ring.current.mol.GetAtomWithIdx(anchor)
    assert atom.GetIsAromatic() and atom.GetAtomicNum() == 6


def _ring_and_methyl(smiles: str):
    tree = fragment(prepare_molecule(smiles, three_d=False))
    ring = select_one(tree, description="phenyl", name="phenyl")
    methyl = select_one(tree, description="methyl", name="methyl", neighbor_of=ring)
    return tree, ring, methyl


def test_resolve_site_para_is_unique_on_benzene():
    tree, ring, methyl = _ring_and_methyl("Cc1ccccc1")
    anchor = attachment_atom(tree, ring, methyl)
    para = resolve_site(tree, ring, methyl, "para", site=is_free_aromatic_carbon)
    dmat_distance = int(Chem.GetDistanceMatrix(ring.current.mol)[anchor][para])
    assert dmat_distance == 3


def test_resolve_site_meta_is_ambiguous_on_benzene():
    tree, ring, methyl = _ring_and_methyl("Cc1ccccc1")
    with pytest.raises(Ambiguous) as info:
        resolve_site(tree, ring, methyl, "meta")
    assert len(info.value.candidates) == 2  # both meta carbons are free


def test_resolve_site_para_rejected_on_five_membered_ring():
    tree = fragment(prepare_molecule("Cc1ccco1", three_d=False))  # 2-methylfuran
    ring = select_one(tree, description="furan", name="furan")
    methyl = select_one(tree, description="methyl", name="methyl", neighbor_of=ring)
    with pytest.raises(ValueError, match="para"):
        resolve_site(tree, ring, methyl, "para")


def test_resolve_position_on_a_chain():
    # "the second carbon of the butyl" — one bond in from the attachment.
    tree = fragment(prepare_molecule("CCCCc1ccccc1", three_d=False))  # butylbenzene
    butyl = select_one(tree, description="butyl", name="butyl")
    anchor = butyl.current.ports[0].anchor_idx
    second = resolve_position(butyl.current.mol, anchor, 1)
    assert butyl.current.mol.GetAtomWithIdx(second).GetTotalNumHs() == 2  # a CH2
