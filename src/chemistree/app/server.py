"""FastAPI web app: 2D + 3D viewers and a command box over one design session.

Configure the shared session with ``state.configure`` before serving (the CLI does
this). Both the command box here and, later, the MCP tools mutate that one session.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from chemistree.app import state
from chemistree.app.commands import run_command
from chemistree.app.render import render_state
from chemistree.app.terminal import terminal_session

app = FastAPI()
_PAGE = (Path(__file__).parent / "index.html").read_text()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """The single-page app."""
    return _PAGE


@app.get("/state")
def current() -> dict:
    """The current viewer state."""
    return render_state(state.get_session())


@app.get("/receptor", response_class=PlainTextResponse)
def receptor() -> str:
    """The receptor PDB block, or empty when no receptor is loaded."""
    return state.receptor_pdb


@app.post("/command")
async def command(payload: dict) -> JSONResponse:
    """Run a command box entry against the shared session, then broadcast."""
    try:
        message = run_command(state.get_session(), payload.get("text", ""))
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    await state.broadcast()
    return JSONResponse(
        {"message": message, "state": render_state(state.get_session())}
    )


@app.websocket("/terminal")
async def terminal(socket: WebSocket) -> None:
    """Bridge an embedded terminal (Claude Code) to a pty."""
    await terminal_session(socket)


@app.websocket("/ws")
async def updates(socket: WebSocket) -> None:
    """Push the current state on connect and on every subsequent change."""
    await socket.accept()
    state.clients.add(socket)
    await socket.send_json(render_state(state.get_session()))
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        state.clients.discard(socket)
