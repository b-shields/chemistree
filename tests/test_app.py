"""App render and command layer (no web server involved)."""

import pytest
from rdkit import Chem

from chemistree import DesignSession
from chemistree.app.commands import run_command
from chemistree.app.render import render_state


def _canonical(smiles: str) -> str:
    return str(Chem.CanonSmiles(smiles))


def test_render_state_has_viewer_artifacts():
    state = render_state(DesignSession("Cc1ccccc1"))
    assert "<svg" in state["svg"]
    assert "V2000" in state["molblock"]  # a molblock with a conformer
    assert state["smiles"] == _canonical("Cc1ccccc1")
    assert "methyl" in state["describe"]


def test_command_swap_uses_a_group_name():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (methyl_id,) = session.find(name="methyl")
    run_command(session, f"swap {methyl_id} trifluoromethyl")
    assert session.smiles() == _canonical("FC(F)(F)c1ccccc1")


def test_command_add_grows_relative_to_reference():
    session = DesignSession("Cc1ccccc1", three_d=False)
    (ring_id,) = session.find(name="phenyl")
    run_command(session, f"add {ring_id} isopropyl para methyl")
    assert session.smiles() == _canonical("CC(C)c1ccc(C)cc1")


def test_command_unknown_raises():
    session = DesignSession("Cc1ccccc1", three_d=False)
    with pytest.raises(ValueError, match="unknown command"):
        run_command(session, "frobnicate 0")
