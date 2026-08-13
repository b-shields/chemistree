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


def test_mutate_between_two_substituents_forms_a_pyridine():
    # m-toluidine: amino and methyl are meta, so one ring carbon sits between
    # them. Turning it into N gives the 2-aminopyridine motif (N next to amino).
    from chemistree import DesignSession

    session = DesignSession("Cc1cccc(N)c1", three_d=False)
    (ring,) = session.find(name="benzene")
    session.mutate(ring, "N", between=("amino", "methyl"))
    assert session.smiles() == str(Chem.CanonSmiles("Cc1cccc(N)n1"))


def test_mutate_ortho_to_a_reference_when_the_other_ortho_is_blocked():
    # o-toluidine: the amino blocks one ortho carbon of the methyl, so
    # "ortho to the methyl" resolves to the single open one. Element by name.
    from chemistree import DesignSession

    session = DesignSession("Cc1ccccc1N", three_d=False)
    (ring,) = session.find(name="benzene")
    session.mutate(ring, "nitrogen", position="ortho", reference="methyl")
    assert session.smiles() == str(Chem.CanonSmiles("Cc1ncccc1N"))
