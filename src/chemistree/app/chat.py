"""Drive Claude Code headless and stream a clean chat feed to the browser.

Each user turn runs ``claude -p`` with stream-json output. The raw stream is
mapped to small UI events (assistant text, a domain-language line per tool call,
end-of-turn) and pushed over a WebSocket. Session continuity is kept with
``--resume`` using the session id Claude reports on the first turn.

The molecule viewers do not update from here. When Claude calls an MCP tool it
mutates the shared session, and the app broadcasts that over ``/ws``. This module
only carries the conversation.
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import WebSocket, WebSocketDisconnect

# The Claude binary and model. Override the binary for a non-default install.
_CLAUDE_BIN = os.environ.get("CHEMISTREE_CLAUDE_BIN", "claude")
_MODEL = "haiku"
# Allow this project's MCP tools without a per-call prompt in headless mode.
_ALLOWED_TOOLS = "mcp__chemistree"
# Block every built-in tool: the agent may edit the molecule via MCP, nothing
# else. It cannot touch the repo, the shell, or the network.
_BLOCKED_TOOLS = ",".join(
    [
        "Bash",
        "BashOutput",
        "KillShell",
        "Edit",
        "Write",
        "Read",
        "NotebookEdit",
        "Glob",
        "Grep",
        "Task",
        "WebFetch",
        "WebSearch",
        "TodoWrite",
        "ExitPlanMode",
    ]
)
# Keep the agent on the molecule and its replies short for the compact feed.
_SYSTEM_PROMPT = (
    "You edit one molecule in a live design session. Use only the chemistree "
    "tools (describe, smiles, find, nearest, swap, add, undo) to inspect and "
    "change it. Never read, write, or run files or shell commands. Reply in one "
    "short sentence."
)

# Tool name -> present-tense phrase shown while the tool runs.
_TOOL_PHRASES = {
    "swap": "swapping the fragment",
    "add": "adding a group",
    "find": "finding the fragment",
    "nearest": "locating the nearest residue",
    "undo": "undoing the last edit",
    "describe": "reading the molecule",
    "smiles": "reading the molecule",
}


def build_command(message: str, session_id: str | None) -> list[str]:
    """Build the ``claude`` argv for one user turn.

    Args:
        message: The user's natural-language request.
        session_id: The id of the running conversation, or None on the first turn.

    Returns:
        The argument list to spawn, resuming the session when one is known.
    """
    cmd = [
        _CLAUDE_BIN,
        "-p",
        message,
        "--model",
        _MODEL,
        "--output-format",
        "stream-json",
        "--verbose",
        "--append-system-prompt",
        _SYSTEM_PROMPT,
        "--allowedTools",
        _ALLOWED_TOOLS,
        "--disallowedTools",
        _BLOCKED_TOOLS,
    ]
    if session_id:
        cmd += ["--resume", session_id]
    return cmd


def tool_phrase(name: str) -> str:
    """Return a domain-language phrase for a tool, or the bare name if unknown.

    Args:
        name: The short tool name (MCP server prefix already stripped).

    Returns:
        A short present-tense status phrase.
    """
    return _TOOL_PHRASES.get(name, name)


def to_events(message: dict) -> list[dict]:
    """Map one raw stream-json message to zero or more UI events.

    Args:
        message: A single decoded object from Claude's stream-json output.

    Returns:
        UI events to send to the browser. A successful result yields nothing:
        its text already arrived as an assistant event, and the driver ends the
        turn on its own.
    """
    kind = message.get("type")
    if kind == "system" and message.get("subtype") == "init":
        return [{"kind": "session", "session_id": message.get("session_id", "")}]
    if kind == "assistant":
        return _assistant_events(message.get("message", {}))
    if kind == "result" and message.get("subtype") != "success":
        detail = message.get("result") or message.get("subtype", "error")
        return [{"kind": "error", "message": detail}]
    return []


def _assistant_events(message: dict) -> list[dict]:
    """Turn an assistant message's content blocks into text and tool events.

    Args:
        message: The ``message`` object of an assistant stream-json line.

    Returns:
        One event per non-empty text block and per tool-use block, in order.
    """
    events: list[dict] = []
    for block in message.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                events.append({"kind": "text", "text": text})
        elif block.get("type") == "tool_use":
            name = _short_tool_name(block.get("name", ""))
            # Show a status line only for molecule edits, not internal plumbing
            # (e.g. the agent's own tool discovery).
            if name in _TOOL_PHRASES:
                events.append(
                    {"kind": "tool", "name": name, "phrase": tool_phrase(name)}
                )
    return events


def _short_tool_name(name: str) -> str:
    """Strip the MCP server prefix, so ``mcp__chemistree__swap`` becomes ``swap``."""
    return name.split("__")[-1]


async def chat_session(socket: WebSocket) -> None:
    """Bridge a browser chat panel to headless Claude Code over a WebSocket.

    Args:
        socket: The accepted WebSocket. Each inbound ``{"type": "message"}`` runs
            one turn; events stream back until the turn ends.
    """
    await socket.accept()
    session_id: str | None = None
    try:
        while True:
            event = json.loads(await socket.receive_text())
            if event.get("type") == "message":
                session_id = await _run_turn(socket, event.get("text", ""), session_id)
    except (WebSocketDisconnect, ValueError):
        pass


async def _run_turn(
    socket: WebSocket, message: str, session_id: str | None
) -> str | None:
    """Run one turn: spawn Claude, stream its events, and end with ``done``.

    Args:
        socket: The chat WebSocket to push events to.
        message: The user's request for this turn.
        session_id: The current session id, or None before the first reply.

    Returns:
        The session id to use next turn (updated from Claude's init event).
    """
    proc = await asyncio.create_subprocess_exec(
        *build_command(message, session_id),
        stdin=asyncio.subprocess.DEVNULL,  # -p reads no stdin; avoid a startup stall
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None

    async for raw in proc.stdout:
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        for ui in to_events(payload):
            if ui["kind"] == "session":
                session_id = ui["session_id"]
            await socket.send_json(ui)

    stderr = (await proc.stderr.read()).decode(errors="ignore").strip()
    if await proc.wait() != 0 and stderr:
        await socket.send_json({"kind": "error", "message": stderr[:500]})
    await socket.send_json({"kind": "done"})
    return session_id
