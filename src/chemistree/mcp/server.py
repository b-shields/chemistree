"""Standalone chemistree MCP server: add it to any agent with ``claude mcp add``.

The server starts empty and holds one design session. The agent calls ``bind`` to
set the molecule (a SMILES string or an SDF path) and an optional receptor, then
edits it with the shared tool set (``tools.register``) — all in-process, no web
app. A fresh server is spawned per stdio connection, so each run is isolated.

Run it directly (``python -m chemistree.mcp.server``) or via the ``chemistree-mcp``
script. ``--tools 2d`` hides the pocket/pose tools; ``--trace <path>`` logs one
JSON record per tool call for benchmarking.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from rdkit import Chem

from chemistree.commands import run_command
from chemistree.mcp import guidance, tools
from chemistree.session import DesignSession

# Env var the chat driver sets so edit results carry the refreshed group listing,
# matching the app's primed mode. app/mcp.py reads the same name.
_PRIME_ENV = "CHEMISTREE_PRIME"
_UNBOUND = "no molecule bound; call bind(molecule) first"


class SessionBackend:
    """Holds one design session and runs commands against it, in-process."""

    def __init__(self, *, prime: bool, trace_path: str | None = None):
        """Start empty; a session appears on the first ``bind``.

        Args:
            prime: Whether edit results carry the refreshed group listing.
            trace_path: Optional JSONL path; when set, every tool call is logged.
        """
        self.prime = prime
        self.session: DesignSession | None = None
        self._trace_path = trace_path

    def bind(self, molecule: str, receptor: str | None = None) -> str:
        """Load a molecule (and optional receptor) into a fresh session.

        Args:
            molecule: A SMILES string or a path to an SDF/MOL file. A file keeps
                its 3D pose; a bare SMILES is embedded only when a receptor makes
                the session 3D.
            receptor: Optional path to a receptor PDB file.

        Returns:
            The new session's group listing, or an ``error:`` message.
        """
        start = time.perf_counter()
        ligand = _load_ligand(molecule)
        if ligand is None:
            return f"error: could not read molecule: {molecule}"
        rec = None
        if receptor is not None:
            rec = Chem.MolFromPDBFile(receptor, removeHs=False, sanitize=False)
            if rec is None:
                return f"error: could not read receptor: {receptor}"
        three_d = rec is not None or _is_posed(ligand)
        try:
            self.session = DesignSession(ligand, rec, three_d=three_d)
        except ValueError as error:
            return f"error: {error}"
        result = self.session.describe()
        self._record("bind", f"{molecule} receptor={receptor}", result, start)
        return result

    def run(self, text: str, *, with_state: bool = False) -> str:
        """Run a text command against the bound session.

        Args:
            text: The command line (see :func:`chemistree.commands.run_command`).
            with_state: Append the refreshed group listing to the result.

        Returns:
            The result message and new SMILES, or an ``error:`` message.
        """
        start = time.perf_counter()
        if self.session is None:
            return f"error: {_UNBOUND}"
        try:
            message = run_command(self.session, text)
        except Exception as error:
            result = f"error: {error}"
        else:
            result = f"{message} | SMILES: {self.session.smiles()}"
            if with_state:
                result += f"\n\nCurrent fragments:\n{self.session.describe()}"
        self._record(text.split()[0], text, result, start)
        return result

    def state(self) -> dict:
        """The current SMILES and group listing, or an unbound message."""
        if self.session is None:
            return {"smiles": _UNBOUND, "describe": _UNBOUND}
        return {"smiles": self.session.smiles(), "describe": self.session.describe()}

    def _record(self, tool: str, args: str, result: str, start: float) -> None:
        """Append one tool-call record to the trace file, if tracing is on."""
        if not self._trace_path:
            return
        record = {
            "ts": time.time(),
            "tool": tool,
            "args": args,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "result_chars": len(result),
            "result_tokens_approx": len(result) // 4,
            "result": result,
            "smiles": self.session.smiles() if self.session else None,
        }
        with open(self._trace_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def _load_ligand(molecule: str) -> str | Chem.Mol | None:
    """A Mol from an existing SDF/MOL path, else the string as a SMILES.

    Returns None only when a file path is given but cannot be read; a bare SMILES
    is passed through unparsed for the session to validate.
    """
    if os.path.exists(molecule):
        return Chem.MolFromMolFile(molecule, removeHs=False)
    return molecule


def _is_posed(ligand: str | Chem.Mol) -> bool:
    """True when a Mol already carries a 3D conformer (a SMILES never does)."""
    return (
        not isinstance(ligand, str)
        and ligand.GetNumConformers() > 0
        and ligand.GetConformer().Is3D()
    )


def build_server(*, profile: str = "all", trace_path: str | None = None):
    """Build the FastMCP server with a ``bind`` tool and the shared tool set.

    Args:
        profile: ``"all"`` or ``"2d"`` (see :func:`chemistree.mcp.tools.register`).
        trace_path: Optional JSONL path for per-tool-call tracing.

    Returns:
        A configured ``FastMCP`` instance.
    """
    from fastmcp import FastMCP

    backend = SessionBackend(
        prime=os.environ.get(_PRIME_ENV) == "1", trace_path=trace_path
    )
    # Server-level guidance so `claude mcp add chemistree` self-describes. Clients
    # surface it to the model as context (soft); the benchmark also delivers it
    # authoritatively via --append-system-prompt.
    mcp = FastMCP("chemistree", instructions=guidance.chemistree_guidance())

    @mcp.tool
    def bind(molecule: str, receptor: str | None = None) -> str:
        """Load the starting molecule to work on, and an optional receptor.

        Call this once, first. ``molecule`` is a SMILES string or a path to an
        SDF/MOL file (a file keeps its 3D pose); ``receptor`` is an optional PDB
        path that turns on the pocket and scoring tools. Returns the group listing.

        This is not an editing tool: to change the molecule, edit it with swap,
        grow, mutate, and remove. Re-binding a hand-written SMILES to apply an edit
        throws away the fragment tree and its ids and skips the ring-position
        feedback and ``matches`` check — so a wrong ring or a misplaced substituent
        goes unnoticed. Bind again only to start over on a different molecule.

        Args:
            molecule: A SMILES string or an SDF/MOL file path.
            receptor: Optional receptor PDB file path.
        """
        return backend.bind(molecule, receptor)

    tools.register(mcp, backend, profile=profile)
    return mcp


def main() -> None:
    """Parse arguments and run the server over stdio."""
    parser = argparse.ArgumentParser(description="Standalone chemistree MCP server.")
    parser.add_argument(
        "--tools",
        choices=("all", "2d"),
        default="all",
        help="'all' (default) or '2d' to hide the pocket/pose tools.",
    )
    parser.add_argument(
        "--trace",
        default=os.environ.get("CHEMISTREE_TRACE"),
        help="JSONL path to log one record per tool call (benchmark mode).",
    )
    args = parser.parse_args()
    build_server(profile=args.tools, trace_path=args.trace).run()


if __name__ == "__main__":
    main()
