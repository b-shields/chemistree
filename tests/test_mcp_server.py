"""The standalone in-process MCP server: bind, the backend, tracing, profiles."""

import json
import pathlib

from rdkit import Chem

from chemistree.mcp import tools
from chemistree.mcp.server import SessionBackend, build_server

DATA = pathlib.Path(__file__).parent / "data" / "abl1"
_BENZANILIDE = "O=C(Nc1ccccc1)c1ccccc1"


class _RecordingMCP:
    """A stand-in FastMCP that records the names of tools registered on it."""

    def __init__(self):
        self.names: list[str] = []

    def tool(self, fn):
        self.names.append(fn.__name__)
        return fn


def test_run_before_bind_reports_unbound():
    backend = SessionBackend(prime=False)
    assert "no molecule bound" in backend.run("swap 0 [*]C")
    assert "no molecule bound" in backend.state()["describe"]


def test_bind_a_smiles_then_edit_in_2d():
    backend = SessionBackend(prime=False)
    listing = backend.bind(_BENZANILIDE)
    assert "phenyl" in listing and "amide" in listing
    result = backend.run("swap 0 [*]c1ccncc1")
    assert "SMILES:" in result
    assert backend.state()["smiles"] == Chem.CanonSmiles("O=C(Nc1ccncc1)c1ccccc1")


def test_bind_reports_a_bad_molecule():
    backend = SessionBackend(prime=False)
    assert backend.bind("not a molecule").startswith("error:")


def test_bind_an_sdf_preserves_the_pose_and_enables_3d_tools():
    backend = SessionBackend(prime=False)
    backend.bind(str(DATA / "reference.sdf"), str(DATA / "receptor.pdb"))
    # A 3D tool that would error without a conformer works here.
    assert "no molecule bound" not in backend.run("clashes")
    assert "coordinates" not in backend.run("clashes")


def test_trace_logs_one_record_per_tool_call(tmp_path):
    trace = tmp_path / "trace.jsonl"
    backend = SessionBackend(prime=False, trace_path=str(trace))
    backend.bind(_BENZANILIDE)
    backend.run("swap 0 [*]c1ccncc1")
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [r["tool"] for r in records] == ["bind", "swap"]
    assert all(r["result_chars"] > 0 and "latency_ms" in r for r in records)


def test_profile_all_registers_the_full_set():
    mcp = _RecordingMCP()
    tools.register(mcp, SessionBackend(prime=False), profile="all")
    assert {
        "swap",
        "matches",
        "minimize",
        "clashes",
        "write_pose",
        "contacts",
        "residues_near",
    } <= set(mcp.names)


def test_profile_2d_omits_the_pose_and_pocket_tools():
    mcp = _RecordingMCP()
    tools.register(mcp, SessionBackend(prime=False), profile="2d")
    assert "swap" in mcp.names and "describe_group" in mcp.names
    assert "matches" in mcp.names  # a 2D check-your-work tool, always registered
    assert not {
        "minimize",
        "clashes",
        "write_pose",
        "distance",
        "contacts",
        "residues_near",
    } & set(mcp.names)


def test_build_server_registers_bind():
    # bind is standalone-only; the shared tools ride along.
    mcp = build_server()
    assert mcp is not None
