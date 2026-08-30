"""Tests for FragmentTree structure operations."""

from rdkit import Chem

from chemistree.session import DesignSession


def _tree():
    """A small fragment tree (p-toluidine) for structural tests."""
    return DesignSession("Cc1ccc(N)cc1", three_d=False).tree


def test_copy_preserves_node_ids_with_distinct_objects():
    tree = _tree()
    snapshot = tree.copy()
    assert [n.id for n in snapshot.nodes] == [n.id for n in tree.nodes]
    for node in tree.nodes:
        # The ids resolve, but to a distinct copy — not the same node object.
        assert snapshot.node(node.id) is not node


def test_copy_is_independent_of_later_edits():
    tree = _tree()
    before = Chem.MolToSmiles(tree.reconstruct())
    snapshot = tree.copy()
    tree.remove_subtree(tree.leaves()[0])  # structurally change the original
    assert Chem.MolToSmiles(tree.reconstruct()) != before  # the original changed
    assert Chem.MolToSmiles(snapshot.reconstruct()) == before  # the snapshot did not
