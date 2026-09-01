"""Remove: prune a node's subtree, capping the parent's port with hydrogen."""

import pytest
from rdkit import Chem

from chemistree import DesignSession


def _canonical(smiles: str) -> str:
    return str(Chem.CanonSmiles(smiles))


def _ring_ids(session: DesignSession) -> list[int]:
    """Ids of the aromatic-ring groups, in tree order."""
    return [
        n.id
        for n in session.tree.nodes
        if n.current.mol.GetRingInfo().NumRings() > 0 and n.id is not None
    ]


def _ring_with_neighbors(session: DesignSession, count: int) -> int:
    """Id of the aromatic ring group with exactly ``count`` neighbours."""
    return next(
        nid
        for nid in _ring_ids(session)
        if len(session.tree.neighbors(session.tree.node(nid))) == count
    )


def test_remove_prunes_a_leaf_and_caps_the_parent():
    # Diphenylamine: remove one phenyl -> aniline (the -NH- keeps its hydrogens).
    session = DesignSession("c1ccccc1Nc1ccccc1", three_d=False)
    session.remove(_ring_with_neighbors(session, 1))  # a terminal phenyl
    assert session.smiles() == _canonical("Nc1ccccc1")


def test_remove_takes_a_rings_substituents_with_it():
    # The para-hydroxy phenyl is a scaffold (it carries an OH leaf). Removing the
    # ring removes its OH too and caps the amine, leaving plain aniline.
    session = DesignSession("c1ccccc1Nc1ccc(O)cc1", three_d=False)
    session.remove(_ring_with_neighbors(session, 2))  # the phenol (amine + hydroxyl)
    assert session.smiles() == _canonical("Nc1ccccc1")


def test_remove_reports_the_capped_grow_position():
    # Diphenylamine: removing one phenyl caps the amine nitrogen with a hydrogen.
    # remove reports (kept group, that hydrogen id) so a replacement can be grown
    # right back at the same spot -- growing a methyl there gives N-methylaniline.
    session = DesignSession("c1ccccc1Nc1ccccc1", three_d=False)
    kept, position = session.remove(_ring_with_neighbors(session, 1))
    session.grow(kept, position, "methyl")
    assert session.smiles() == _canonical("CNc1ccccc1")


def test_remove_can_be_undone():
    # Deleting a ring and reverting it must restore the molecule exactly.
    session = DesignSession("c1ccccc1Nc1ccc(O)cc1", three_d=False)
    original = session.smiles()
    session.remove(_ring_with_neighbors(session, 2))
    assert session.smiles() != original
    session.undo()
    assert session.smiles() == original


def test_remove_rejects_the_only_fragment():
    session = DesignSession("c1ccccc1", three_d=False)  # a single-node tree
    (only_id,) = (n.id for n in session.tree.nodes)
    with pytest.raises(ValueError):
        session.remove(only_id)
