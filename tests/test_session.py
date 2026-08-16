"""The redesigned, id-addressed design session."""

import pathlib
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from chemistree.fragmenter import fragment
from chemistree.session import DesignSession

DATA = pathlib.Path(__file__).parent / "data" / "abl1"


def _abl1_session() -> DesignSession:
    """A posed abl1 ligand with its receptor, for spatial tests."""
    ligand = Chem.MolFromMolFile(str(DATA / "reference.sdf"), removeHs=False)
    receptor = Chem.MolFromPDBFile(
        str(DATA / "receptor.pdb"), removeHs=False, sanitize=False
    )
    return DesignSession(ligand, receptor)


def _fragment_multiset(mol_or_tree) -> Counter:
    """Canonical fragment SMILES (ports as bare `*`) of a tree or a fresh mol."""
    tree = mol_or_tree if hasattr(mol_or_tree, "nodes") else fragment(mol_or_tree)
    smiles = []
    for node in tree.nodes:
        m = Chem.Mol(node.current.mol)
        for atom in m.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)  # ignore port labels; compare shapes only
        smiles.append(Chem.MolToSmiles(Chem.RemoveHs(m)))
    return Counter(smiles)


def _built_from(session) -> Counter:
    """The fragment multiset of constructing fresh from the current molecule."""
    return _fragment_multiset(session.molecule())


def test_describe_is_a_compact_overview_without_per_group_detail():
    session = DesignSession("Cc1ccc(N)cc1", three_d=False)
    text = session.describe()
    assert text.startswith("# Group Summary")
    assert text.count("\n- [") == len(session.tree.nodes)  # one bullet per group
    # The heavy per-group detail is pulled on demand, not dumped here.
    assert "**Atom Map:**" not in text
    assert "**Topology:**" not in text


def test_describe_group_shows_atom_map_topology_and_rings():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    methyl = next(
        n.id
        for n in session.tree.nodes
        if sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == 1
    )
    ring_detail = session.describe_group(ring)
    assert "**Atom Map:**" in ring_detail
    assert "**Positions:**" in ring_detail
    assert "**Rings:**" in ring_detail
    assert "position_id" in ring_detail  # the legend lives with the detail
    assert "**Rings:**" not in session.describe_group(methyl)  # acyclic
    # The raw matrix is available behind the flag for comparison.
    assert "**Topology:**" in session.describe_group(ring, use_matrix=True)


def test_smiles_history_records_construction():
    session = DesignSession("Cc1ccccc1", three_d=False)
    assert session.smiles_history == [Chem.CanonSmiles("Cc1ccccc1")]


def _aromatic_h_on_ring(session, node_id):
    """An id of a hydrogen on the aromatic ring of the given node."""
    mol = session.tree.node(node_id).current.mol
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetIsAromatic():
            return atom.GetIdx()
    raise AssertionError("no aromatic hydrogen found")


def _methyl_h(session, node_id):
    """An id of a hydrogen on a (single-carbon) methyl node."""
    mol = session.tree.node(node_id).current.mol
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)


def test_swap_obeys_the_edit_construct_invariant():
    session = DesignSession("Cc1ccccc1", three_d=False)
    # swap the methyl leaf for a benzyl group, which itself splits at construction
    leaf = next(
        n.id
        for n in session.tree.nodes
        if sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == 1
    )
    session.swap(leaf, "[*]Cc1ccccc1")
    assert _fragment_multiset(session.tree) == _built_from(session)


def test_grow_split_case_adds_a_new_node_on_a_ring():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    before = len(session.tree.nodes)
    grown = session.grow(ring, _aromatic_h_on_ring(session, ring), "methyl")
    assert session.smiles() == Chem.CanonSmiles("Cc1ccccc1C")  # an o-/m-/p-xylene
    assert len(session.tree.nodes) == before + 1  # a new leaf node
    assert (
        grown not in {n for n in range(before)} or grown == session.tree.node(grown).id
    )
    assert _fragment_multiset(session.tree) == _built_from(session)


def test_grow_merge_case_extends_the_fragment_without_a_new_node():
    session = DesignSession("Cc1ccccc1", three_d=False)
    methyl = next(
        n.id
        for n in session.tree.nodes
        if sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == 1
    )
    before = len(session.tree.nodes)
    session.grow(methyl, _methyl_h(session, methyl), "methyl")
    assert session.smiles() == Chem.CanonSmiles("CCc1ccccc1")  # ethylbenzene
    assert len(session.tree.nodes) == before  # merged: no new node
    assert _fragment_multiset(session.tree) == _built_from(session)


def test_grow_rejects_a_heavy_atom_position():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    with pytest.raises(ValueError, match="hydrogen"):
        session.grow(ring, 0, "methyl")  # atom 0 is a heavy ring carbon


def test_mutate_ring_carbon_to_nitrogen():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    # a ring carbon that bears a hydrogen (not the one carrying the methyl port)
    mol = session.tree.node(ring).current.mol
    carbon = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetTotalNumHs(includeNeighbors=True) == 1
    )
    session.mutate(ring, carbon, "N")
    assert session.smiles() == Chem.CanonSmiles("Cc1ccccn1")  # a methylpyridine


def test_mutate_rejects_a_hydrogen_position():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    with pytest.raises(ValueError, match="heavy atom"):
        session.mutate(ring, _aromatic_h_on_ring(session, ring), "N")


def test_mutate_invalid_element_raises_descriptive_error():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    mol = session.tree.node(ring).current.mol
    carbon = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetTotalNumHs(includeNeighbors=True) == 1
    )
    with pytest.raises(ValueError, match="aromatic"):
        session.mutate(ring, carbon, "O")  # oxygen cannot hold the aromatic ring


def test_grow_in_3d_keeps_bond_lengths_sane():
    # A warped placement shows up as an over-long heavy bond or a collapsed C-H.
    session = DesignSession("Cc1ccccc1", three_d=True)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    session.grow(ring, _aromatic_h_on_ring(session, ring), "ethyl")
    conf = session.molecule().GetConformer()
    heavy, ch = [], []
    for bond in session.molecule().GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        d = conf.GetAtomPosition(a.GetIdx()).Distance(conf.GetAtomPosition(b.GetIdx()))
        if a.GetAtomicNum() > 1 and b.GetAtomicNum() > 1:
            heavy.append(d)
        if {a.GetAtomicNum(), b.GetAtomicNum()} == {6, 1}:
            ch.append(d)
    assert max(heavy) < 1.7
    assert min(ch) > 1.0


def test_2d_only_grow_and_mutate_need_no_conformer():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    session.grow(ring, _aromatic_h_on_ring(session, ring), "[*]F")
    assert session.smiles() == Chem.CanonSmiles("Cc1ccccc1F")
    assert "**Atom Map:**" in session.describe_group(ring)


def test_distance_report_matches_ground_truth_minimum():
    session = _abl1_session()
    report = session.distance("ASP")
    assert report.startswith("# Distances to ASP")
    assert "## Details" in report

    # The smallest distance printed must equal the true closest ligand-ASP approach.
    residue = session.receptor.residue_atoms("ASP")
    truth = min(
        float(
            np.sqrt(
                ((residue - np.array(list(conf.GetAtomPosition(a.GetIdx())))) ** 2).sum(
                    -1
                )
            ).min()
        )
        for node in session.tree.nodes
        for conf in [node.current.mol.GetConformer()]
        for a in node.current.mol.GetAtoms()
        if a.GetAtomicNum() > 1
    )
    printed = min(
        float(token)
        for line in report.splitlines()
        if line.startswith("| [")
        for token in [line.split("|")[2].strip()]
    )
    assert printed == pytest.approx(truth, abs=0.01)


def test_distance_requires_a_receptor():
    session = DesignSession("Cc1ccccc1", three_d=True)
    with pytest.raises(ValueError, match="receptor"):
        session.distance("ASP")


def test_edits_are_undoable():
    session = DesignSession("Cc1ccccc1", three_d=False)
    start = session.smiles()
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    session.grow(ring, _aromatic_h_on_ring(session, ring), "methyl")
    session.undo()
    assert session.smiles() == start
