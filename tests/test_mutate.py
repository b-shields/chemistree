"""Atom mutation: change one atom's element in place (e.g. aromatic C to N)."""

import pytest
from rdkit import Chem

from chemistree import fragment
from chemistree.edits import mutate_atom


def _product(tree) -> str:
    """Canonical SMILES of a tree's reconstructed molecule, Hs removed."""
    return str(Chem.MolToSmiles(Chem.RemoveHs(tree.reconstruct())))


def test_mutate_benzene_carbon_to_nitrogen_gives_pyridine():
    tree = fragment(Chem.MolFromSmiles("c1ccccc1"))
    (ring,) = tree.nodes
    mutate_atom(ring, 0, 7)  # carbon -> nitrogen
    assert _product(tree) == str(Chem.CanonSmiles("c1ccncc1"))


def test_mutate_to_an_impossible_valence_raises():
    tree = fragment(Chem.MolFromSmiles("Cc1ccccc1"))
    ring = next(
        node
        for node in tree.nodes
        if any(a.GetIsAromatic() for a in node.current.mol.GetAtoms())
    )
    # The ipso carbon (index of the port's neighbor) already bears a substituent;
    # turning an aromatic carbon into oxygen cannot satisfy valence.
    with pytest.raises(ValueError):
        mutate_atom(ring, 1, 8)  # -> oxygen in an aromatic ring
