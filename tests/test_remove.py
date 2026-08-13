"""Remove: prune a node's subtree, capping the parent's port with hydrogen."""

import pytest
from rdkit import Chem

from chemistree import DesignSession


def _canonical(smiles: str) -> str:
    return str(Chem.CanonSmiles(smiles))


def test_remove_prunes_a_leaf_and_caps_the_parent():
    # Diphenylamine: remove one phenyl -> aniline (the -NH- keeps its hydrogens).
    session = DesignSession("c1ccccc1Nc1ccccc1", three_d=False)
    phenyl_id = session.find(name="phenyl")[0]
    session.remove(phenyl_id)
    assert session.smiles() == _canonical("Nc1ccccc1")


def test_remove_takes_a_rings_substituents_with_it():
    # The para-hydroxy phenyl is a scaffold (it carries an OH leaf). Removing the
    # ring removes its OH too and caps the amine, leaving plain aniline.
    session = DesignSession("c1ccccc1Nc1ccc(O)cc1", three_d=False)
    # The phenol ring carries the OH, so it has two neighbors (amine + hydroxyl);
    # the plain phenyl has one. Pick the two-neighbor benzene.
    phenol = next(
        nid
        for nid in session.find(name="benzene")
        if len(session.tree.neighbors(session.tree.node(nid))) == 2
    )
    session.remove(phenol)
    assert session.smiles() == _canonical("Nc1ccccc1")


def test_remove_can_be_undone():
    # Deleting a ring and reverting it must restore the molecule exactly.
    session = DesignSession("c1ccccc1Nc1ccc(O)cc1", three_d=False)
    original = session.smiles()
    phenol = next(
        nid
        for nid in session.find(name="benzene")
        if len(session.tree.neighbors(session.tree.node(nid))) == 2
    )
    session.remove(phenol)
    assert session.smiles() != original
    session.undo()
    assert session.smiles() == original


def test_remove_rejects_the_only_fragment():
    session = DesignSession("c1ccccc1", three_d=False)  # a single-node tree
    (only_id,) = (n.id for n in session.tree.nodes)
    with pytest.raises(ValueError):
        session.remove(only_id)


def test_fill_grows_a_group_where_the_last_remove_freed_a_site():
    # Remove a phenyl from diphenylamine (-> aniline, freeing the N's site),
    # then fill that site with a methyl -> N-methylaniline.
    session = DesignSession("c1ccccc1Nc1ccccc1", three_d=False)
    session.remove(session.find(name="phenyl")[0])
    session.fill("methyl")
    assert session.smiles() == _canonical("CNc1ccccc1")


def test_fill_can_be_undone():
    session = DesignSession("c1ccccc1Nc1ccccc1", three_d=False)
    session.remove(session.find(name="phenyl")[0])
    aniline = session.smiles()
    session.fill("methyl")
    session.undo()
    assert session.smiles() == aniline
