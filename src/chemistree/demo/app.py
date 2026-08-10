"""FastAPI web demo: 2D + 3D viewers and a command box over one design session.

The session is a process-global so it persists across requests and, later, so the
MCP tools and the command box can drive the same state. Run with::

    python -m chemistree.demo
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from chemistree.demo.commands import run_command
from chemistree.demo.render import render_state
from chemistree.session import DesignSession

app = FastAPI()

# One shared molecule for the whole demo (aspirin to start).
session = DesignSession("CC(=O)Oc1ccccc1C(=O)O")
_clients: set[WebSocket] = set()
_PAGE = (Path(__file__).parent / "index.html").read_text()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """The single-page demo."""
    return _PAGE


@app.get("/state")
def state() -> dict:
    """The current viewer state."""
    return render_state(session)


@app.post("/command")
async def command(payload: dict) -> JSONResponse:
    """Run a command box entry against the shared session, then broadcast."""
    try:
        message = run_command(session, payload.get("text", ""))
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    await _broadcast()
    return JSONResponse({"message": message, "state": render_state(session)})


@app.websocket("/ws")
async def updates(socket: WebSocket) -> None:
    """Push the current state on connect and on every subsequent change."""
    await socket.accept()
    _clients.add(socket)
    await socket.send_json(render_state(session))
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        _clients.discard(socket)


async def _broadcast() -> None:
    """Send the current state to every connected viewer."""
    current = render_state(session)
    for client in list(_clients):
        try:
            await client.send_json(current)
        except Exception:
            _clients.discard(client)
