"""FastAPI web app: 2D + 3D viewers and a command box over one design session.

Configure the shared session with ``state.configure`` before serving (the CLI does
this). Both the command box here and, later, the MCP tools mutate that one session.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from chemistree.app import state, voice
from chemistree.app.chat import chat_session
from chemistree.app.render import render_molecule, render_state
from chemistree.commands import run_command

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


@app.get("/step/{index}")
def step(index: int) -> JSONResponse:
    """Viewer artifacts for one molecule in the trace (a filmstrip thumbnail click).

    Args:
        index: Position in the session history, 0 = the starting molecule.

    Returns:
        The step's SMILES, 2D SVG, and 3D molblock, or a 404 error for a bad index.
    """
    history = state.get_session().history()
    if not 0 <= index < len(history):
        return JSONResponse({"error": "no such step"}, status_code=404)
    return JSONResponse(render_molecule(history[index]))


@app.get("/receptor", response_class=PlainTextResponse)
def receptor() -> str:
    """The receptor PDB block, or empty when no receptor is loaded."""
    return state.receptor_pdb


@app.get("/pocket")
def pocket() -> list[dict]:
    """Binding-site residues (within 6 A of the ligand) to render as lines."""
    session = state.get_session()
    if session.receptor is None:
        return []
    return [
        {"chain": residue.chain, "resi": residue.number, "resn": residue.name}
        for residue in session.receptor.pocket(session.molecule(), within=6.0)
    ]


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


@app.websocket("/chat")
async def chat(socket: WebSocket) -> None:
    """Bridge the chat panel to headless Claude Code in the configured mode."""
    mode = state.get_mode()
    # Seed the current fragments (at connect, so a reload after edits is current).
    context = state.get_session().describe() if mode.prime_context else ""
    await chat_session(socket, mode, context, state.get_model(), state.get_narrate())


@app.websocket("/voice")
async def voice_input(socket: WebSocket) -> None:
    """Stream transcribed speech while the socket is open (the mic toggle)."""
    await socket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()
    stop = threading.Event()

    def on_text(text: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, text)

    def run() -> None:
        try:
            voice.listen(on_text, stop)
        except Exception as error:  # missing voice deps, or no microphone
            loop.call_soon_threadsafe(queue.put_nowait, f"__error__:{error}")

    threading.Thread(target=run, daemon=True).start()
    sender = asyncio.create_task(_send_transcripts(socket, queue))
    receiver = asyncio.create_task(_watch_close(socket))
    try:
        await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        sender.cancel()
        receiver.cancel()


async def _send_transcripts(socket: WebSocket, queue: asyncio.Queue[str]) -> None:
    """Forward queued transcripts (or a one-off error) to the client."""
    while True:
        text = await queue.get()
        if text.startswith("__error__:"):
            await socket.send_json({"type": "error", "message": text[10:]})
            return
        await socket.send_json({"type": "transcript", "text": text})


async def _watch_close(socket: WebSocket) -> None:
    """Resolve when the client closes the socket (mic toggled off)."""
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        return


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
