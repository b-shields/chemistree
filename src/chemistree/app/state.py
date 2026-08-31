"""Shared application state: the current design session and connected viewers.

The session is a process-global so the web routes and (later) the MCP tools drive
the same molecule, and so the viewers can be pushed updates over a WebSocket.
"""

from __future__ import annotations

from fastapi import WebSocket
from rdkit import Chem

from chemistree.app.chat import DEFAULT_MODE, DEFAULT_MODEL, ChatMode
from chemistree.app.render import render_state
from chemistree.session import DesignSession

_session: DesignSession | None = None
_mode: ChatMode = DEFAULT_MODE
_model: str = DEFAULT_MODEL
_narrate: bool = True
receptor_pdb: str = ""
clients: set[WebSocket] = set()


def configure(
    ligand: Chem.Mol,
    receptor: Chem.Mol | None = None,
    *,
    mode: ChatMode = DEFAULT_MODE,
    model: str = DEFAULT_MODEL,
    narrate: bool = True,
) -> None:
    """Set the molecule (and optional receptor) the app edits.

    Args:
        ligand: The ligand molecule, posed in 3D when a receptor is given.
        receptor: Optional receptor for proximity context and 3D display.
        mode: The chat mode the ``/chat`` endpoint runs in.
        model: The Claude model the ``/chat`` endpoint runs.
        narrate: Whether the agent narrates a one-line rationale per edit.
    """
    global _session, _mode, _model, _narrate, receptor_pdb
    _session = DesignSession(ligand, receptor)
    _mode = mode
    _model = model
    _narrate = narrate
    receptor_pdb = Chem.MolToPDBBlock(receptor) if receptor is not None else ""


def get_mode() -> ChatMode:
    """The chat mode selected at configuration time."""
    return _mode


def get_model() -> str:
    """The Claude model selected at configuration time."""
    return _model


def get_narrate() -> bool:
    """Whether the agent narrates its reasoning, selected at configuration time."""
    return _narrate


def get_session() -> DesignSession:
    """The configured session.

    Returns:
        The current design session.

    Raises:
        RuntimeError: If the app has not been configured yet.
    """
    if _session is None:
        raise RuntimeError("app is not configured; call configure() first")
    return _session


async def broadcast() -> None:
    """Send the current state to every connected viewer."""
    current = render_state(get_session())
    for client in list(clients):
        try:
            await client.send_json(current)
        except Exception:
            clients.discard(client)
