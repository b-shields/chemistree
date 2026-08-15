"""DesignSession: the stateful facade over fragment / selection / edits."""

import pytest
from rdkit import Chem

from chemistree import DesignSession


def _canonical(smiles: str) -> str:
    return str(Chem.CanonSmiles(smiles))


def test_session_reconstructs_the_input():
    session = DesignSession("Cc1ccccc1", three_d=False)
    assert session.smiles() == _canonical("Cc1ccccc1")


def test_find_returns_node_ids_by_name():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (methyl_id,) = session.find(name="methyl")
    assert isinstance(methyl_id, int)


def test_swap_accepts_a_group_name():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (methyl_id,) = session.find(name="methyl")
    session.swap(methyl_id, "trifluoromethyl")  # name, not SMILES
    assert session.smiles() == _canonical("FC(F)(F)c1ccccc1")


def test_add_grows_relative_to_a_named_reference():
    # "add an isopropyl para to the methyl on the benzene" -> p-cymene.
    session = DesignSession("Cc1ccccc1", three_d=False)
    (ring_id,) = session.find(name="phenyl")
    session.add(ring_id, "isopropyl", position="para", reference="methyl")
    assert session.smiles() == _canonical("CC(C)c1ccc(C)cc1")


def test_undo_reverts_the_last_edit():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (methyl_id,) = session.find(name="methyl")
    session.swap(methyl_id, "[*]Cl")
    session.undo()
    assert session.smiles() == _canonical("Cc1ccccc1")


def test_undo_walks_back_edits_in_order():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (methyl_id,) = session.find(name="methyl")
    session.swap(methyl_id, "trifluoromethyl")
    (ring_id,) = session.find(name="phenyl")
    session.add(ring_id, "chloro", position="para", reference="trifluoromethyl")
    session.undo()  # reverts the add
    assert session.smiles() == _canonical("FC(F)(F)c1ccccc1")
    session.undo()  # reverts the swap
    assert session.smiles() == _canonical("Cc1ccccc1")


def test_describe_and_annotations_reflect_state():
    session = DesignSession("Cc1ccccc1", three_d=False)
    assert "methyl" in session.describe()
    names = {node["name"] for node in session.annotations()["nodes"]}
    assert names == {"methyl", "phenyl"}


def test_swap_unknown_group_raises_expressive_error():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (methyl_id,) = session.find(name="methyl")
    with pytest.raises(ValueError, match="unknown group 'flibberto'"):
        session.swap(methyl_id, "flibberto")


def test_add_resolves_symmetric_references_to_their_shared_site():
    # 2,6-dichlorotoluene: "meta to the Cl" is the same open position for both
    # chlorines (para to the methyl), so it is not ambiguous — pick it.
    session = DesignSession("Cc1c(Cl)cccc1Cl", three_d=False)
    (ring,) = session.find(name="benzene")
    session.add(ring, "tert-butyl", position="meta", reference="chloro")
    assert session.smiles() == _canonical("Cc1c(Cl)cc(C(C)(C)C)cc1Cl")


def test_smiles_history_seeds_with_the_input():
    session = DesignSession("Cc1ccccc1", three_d=False)
    assert session.smiles_history == [_canonical("Cc1ccccc1")]


def test_smiles_history_records_each_edit_in_order():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (methyl_id,) = session.find(name="methyl")
    session.swap(methyl_id, "trifluoromethyl")
    (ring_id,) = session.find(name="phenyl")
    session.add(ring_id, "chloro", position="para", reference="trifluoromethyl")
    session.undo()
    assert session.smiles_history == [
        _canonical("Cc1ccccc1"),
        _canonical("FC(F)(F)c1ccccc1"),
        _canonical("FC(F)(F)c1ccc(Cl)cc1"),
        _canonical("FC(F)(F)c1ccccc1"),
    ]
