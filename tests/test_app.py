"""App render and command layer (no web server involved)."""

import pytest
from rdkit import Chem

from chemistree import DesignSession
from chemistree.app.commands import run_command
from chemistree.app.render import render_state


def _canonical(smiles: str) -> str:
    return str(Chem.CanonSmiles(smiles))


def _heavy(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _node_id(session: DesignSession, predicate) -> int:
    """The id of the single node matching a predicate on its current fragment."""
    node = next(n for n in session.tree.nodes if predicate(n.current.mol))
    assert node.id is not None
    return node.id


def _aromatic_h(session: DesignSession, node_id: int) -> int:
    """An id of a hydrogen on the aromatic ring of the given node."""
    mol = session.tree.node(node_id).current.mol
    return int(
        next(
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetIsAromatic()
        )
    )


def test_render_state_has_viewer_artifacts():
    state = render_state(DesignSession("Cc1ccccc1"))
    assert "<svg" in state["svg"]
    assert "V2000" in state["molblock"]  # a molblock with a conformer
    assert state["smiles"] == _canonical("Cc1ccccc1")
    assert "methyl" in state["describe"]


def test_render_state_trace_mirrors_the_edit_history():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring_id = _node_id(session, lambda m: m.GetRingInfo().NumRings() > 0)
    run_command(session, f"grow {ring_id} {_aromatic_h(session, ring_id)} methyl")
    trace = render_state(session)["trace"]
    assert [entry["smiles"] for entry in trace] == session.smiles_history
    assert "<svg" in trace[-1]["svg"]  # a thumbnail per history step
    assert trace[-1]["smiles"] == render_state(session)["smiles"]  # newest is current


def test_trace_affinity_is_none_without_a_receptor():
    trace = render_state(DesignSession("Cc1ccccc1", three_d=False))["trace"]
    assert all(entry["affinity"] is None for entry in trace)


def test_trace_carries_a_vinardo_number_per_step_with_a_receptor():
    import pathlib

    data = pathlib.Path(__file__).parent / "data" / "abl1"
    ligand = Chem.MolFromMolFile(str(data / "reference.sdf"), removeHs=False)
    receptor = Chem.MolFromPDBFile(
        str(data / "receptor.pdb"), removeHs=False, sanitize=False
    )
    trace = render_state(DesignSession(ligand, receptor))["trace"]
    assert [entry["affinity"] for entry in trace] == pytest.approx([-11.6], abs=0.2)


def test_command_swap_uses_a_group_name():
    session = DesignSession("Cc1ccccc1", three_d=False)
    methyl_id = _node_id(session, lambda m: _heavy(m) == 1)
    run_command(session, f"swap {methyl_id} trifluoromethyl")
    assert session.smiles() == _canonical("FC(F)(F)c1ccccc1")


def test_command_grow_adds_a_group_at_a_position():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring_id = _node_id(session, lambda m: m.GetRingInfo().NumRings() > 0)
    run_command(session, f"grow {ring_id} {_aromatic_h(session, ring_id)} methyl")
    assert session.smiles() == _canonical("Cc1ccccc1C")  # a xylene


def test_command_mutate_changes_an_atom():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring_id = _node_id(session, lambda m: m.GetRingInfo().NumRings() > 0)
    mol = session.tree.node(ring_id).current.mol
    carbon = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetTotalNumHs(includeNeighbors=True) == 1
    )
    run_command(session, f"mutate {ring_id} {carbon} N")
    assert session.smiles() == _canonical("Cc1ccccn1")  # a methylpyridine


def test_command_group_returns_the_detail_section():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring_id = _node_id(session, lambda m: m.GetRingInfo().NumRings() > 0)
    detail = run_command(session, f"group {ring_id}")
    assert "**Positions:**" in detail


def test_command_contacts_reports_site_contacts():
    import pathlib

    data = pathlib.Path(__file__).parent / "data" / "abl1"
    ligand = Chem.MolFromMolFile(str(data / "reference.sdf"), removeHs=False)
    receptor = Chem.MolFromPDBFile(
        str(data / "receptor.pdb"), removeHs=False, sanitize=False
    )
    session = DesignSession(ligand, receptor)
    report = run_command(session, "contacts 4.5")
    assert report.startswith("# Binding-site contacts (within 4.5 A)")


def test_command_minimize_returns_a_report():
    session = DesignSession("CCc1ccccc1", three_d=True)  # ethylbenzene
    ethyl = _node_id(session, lambda m: _heavy(m) == 2)
    report = run_command(session, f"minimize {ethyl}")
    assert report.startswith("Settled")
    assert "Energy" in report


def test_command_clashes_returns_a_report():
    session = DesignSession("Cc1ccccc1", three_d=True)
    assert "Clashes" in run_command(session, "clashes")


def test_command_unknown_raises():
    session = DesignSession("Cc1ccccc1", three_d=False)
    with pytest.raises(ValueError, match="unknown command"):
        run_command(session, "frobnicate 0")
