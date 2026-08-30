"""Drive Claude Code headless and stream a clean chat feed to the browser.

One persistent ``claude`` process serves the whole conversation. It runs in
stream-json input mode: each user turn is written to its stdin as one JSON line,
and its stdout is read until the turn's ``result`` message. Holding the process
open keeps the model warm — only the first turn pays cold-start; later turns
skip the process launch, the MCP handshake, and the session replay.

The raw stream is mapped to small UI events (assistant text, a domain-language
line per tool call, end-of-turn) and pushed over a WebSocket.

The molecule viewers do not update from here. When Claude calls an MCP tool it
mutates the shared session, and the app broadcasts that over ``/ws``. This module
only carries the conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

_log = logging.getLogger(__name__)

# The Claude binary and default model. Override the binary for a non-default
# install; the model is chosen per run by the ``--model`` CLI flag.
_CLAUDE_BIN = os.environ.get("CHEMISTREE_CLAUDE_BIN", "claude")
DEFAULT_MODEL = "haiku"
# The editable prompt template, read once. It carries {tools} and {primed_note}
# placeholders that _system_prompt fills per mode.
_PROMPT_TEMPLATE = (Path(__file__).parent / "system_prompt.md").read_text()
# Env var the chat driver sets so the MCP server (a child of the claude process)
# attaches the refreshed fragment listing to edit results. app/mcp.py reads the
# same name.
_PRIME_ENV = "CHEMISTREE_PRIME"
# Allow this project's MCP tools without a per-call prompt in headless mode.
_ALLOWED_TOOLS = "mcp__chemistree"
# Load only this project's MCP server. Without strict scoping Claude also loads
# the user's global servers (Gmail/Drive/Calendar) every turn, which the demo
# never uses; they add startup cost and auth prompts. The app runs from the repo
# root, so the relative path resolves.
_MCP_CONFIG = ".mcp.json"
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
def _system_prompt(tools: str, primed_note: str = "") -> str:
    """Build the agent's system prompt naming the tools a mode exposes.

    Fills the ``system_prompt.md`` template so the prompt text stays editable in
    plain markdown, not inline in this module.

    Args:
        tools: Comma-separated tool names the mode exposes, named in the prompt.
        primed_note: Extra guidance appended for the primed mode.

    Returns:
        The full ``--append-system-prompt`` text.
    """
    return _PROMPT_TEMPLATE.format(tools=tools, primed_note=primed_note).rstrip()


_SYSTEM_PROMPT = _system_prompt(
    "describe, describe_group, smiles, swap, grow, mutate, remove, rotate, undo, "
    "distance, contacts, clashes"
)
# The primed mode seeds the group listing up front and refreshes it after each edit,
# so the describe overview is blocked; the agent acts on the given ids and calls
# describe_group for atom positions.
_PRIMED_PROMPT = _system_prompt(
    "describe_group, smiles, swap, grow, mutate, remove, rotate, undo, distance, "
    "contacts, clashes",
    primed_note=(
        "# Session note\n\n"
        "You are given the current group listing, refreshed after every edit; use "
        "those ids directly and call describe_group(id) for atom positions."
    ),
)


@dataclass(frozen=True)
class ChatMode:
    """A chat experience: its system prompt and whether it primes state.

    Attributes:
        name: Selector used by the ``--chat-mode`` CLI flag.
        system_prompt: The full ``--append-system-prompt`` text for this mode.
        prime_context: Seed the fragment listing at connect and refresh it in
            every edit result, so the agent skips find/describe lookups.
        blocked_tools: chemistree tool names to hide in this mode (short names,
            e.g. "find"). Empty for explore; primed blocks the lookups priming
            makes redundant.
    """

    name: str
    system_prompt: str
    prime_context: bool
    blocked_tools: tuple[str, ...] = ()


EXPLORE = ChatMode(name="explore", system_prompt=_SYSTEM_PROMPT, prime_context=False)
PRIMED = ChatMode(
    name="primed",
    system_prompt=_PRIMED_PROMPT,
    prime_context=True,
    blocked_tools=("describe",),
)
MODES: dict[str, ChatMode] = {EXPLORE.name: EXPLORE, PRIMED.name: PRIMED}
DEFAULT_MODE = PRIMED

# Tool name -> present-tense phrase shown while the tool runs.
_TOOL_PHRASES = {
    "swap": "swapping the fragment",
    "grow": "growing a group",
    "mutate": "mutating an atom",
    "remove": "removing a group",
    "rotate": "rotating a group",
    "undo": "undoing the last edit",
    "distance": "measuring distances",
    "contacts": "mapping the binding site",
    "clashes": "checking for clashes",
    "describe": "reading the molecule",
    "describe_group": "reading a group",
    "smiles": "reading the molecule",
}


def build_command(
    mode: ChatMode, context: str = "", model: str = DEFAULT_MODEL
) -> list[str]:
    """Build the ``claude`` argv for the persistent streaming session.

    The process is spawned once and reused for every turn. It reads user
    messages from stdin as stream-json and writes stream-json to stdout, so no
    per-turn message or ``--resume`` id is needed here.

    Args:
        mode: The chat mode, which chooses the system prompt.
        context: The current fragment listing to seed. Used only when the mode
            primes state; ignored otherwise.
        model: The Claude model alias to run (haiku, sonnet, opus, or a full id).

    Returns:
        The argument list to spawn.
    """
    prompt = mode.system_prompt
    if mode.prime_context and context:
        prompt += f"\n\nCurrent fragments (use these node ids directly):\n{context}"
    # Block the built-in tools always, plus this mode's redundant MCP lookups.
    disallowed = _BLOCKED_TOOLS
    for tool in mode.blocked_tools:
        disallowed += f",{_ALLOWED_TOOLS}__{tool}"
    return [
        _CLAUDE_BIN,
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--append-system-prompt",
        prompt,
        "--mcp-config",
        _MCP_CONFIG,
        "--strict-mcp-config",
        "--allowedTools",
        _ALLOWED_TOOLS,
        "--disallowedTools",
        disallowed,
    ]


def user_message_line(text: str) -> bytes:
    """Encode a user turn as one stream-json input line.

    Args:
        text: The user's natural-language request.

    Returns:
        The newline-terminated JSON line to write to the process's stdin.
    """
    message = {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }
    return (json.dumps(message) + "\n").encode()


def interrupt_line() -> bytes:
    """Encode an interrupt control request as one stream-json input line.

    Returns:
        The newline-terminated JSON line that aborts the in-flight turn when
        written to the process's stdin, leaving the process warm for the next turn.
    """
    message = {
        "type": "control_request",
        "request_id": uuid.uuid4().hex,
        "request": {"subtype": "interrupt"},
    }
    return (json.dumps(message) + "\n").encode()


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
            # Log every tool call (including hidden ones) so the steps we combine
            # away in primed mode stay visible for debugging.
            _log.debug("tool_use: %s", name)
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


async def chat_session(
    socket: WebSocket,
    mode: ChatMode = DEFAULT_MODE,
    context: str = "",
    model: str = DEFAULT_MODEL,
) -> None:
    """Bridge a browser chat panel to one persistent Claude Code process.

    Spawns the process once, then feeds each inbound ``{"type": "message"}`` to
    its stdin and streams the reply back until the turn ends. The process is
    reused across turns and torn down when the socket closes.

    Args:
        socket: The accepted WebSocket.
        mode: The chat mode, which chooses the system prompt and whether to prime.
        context: The current fragment listing to seed when the mode primes.
        model: The Claude model alias to run.
    """
    await socket.accept()
    # In the primed mode the MCP server must attach the listing to edit results;
    # it is a child of this process, so pass the flag down through the environment.
    env = {**os.environ, _PRIME_ENV: "1"} if mode.prime_context else None
    proc = await asyncio.create_subprocess_exec(
        *build_command(mode, context, model),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert proc.stderr is not None
    drain = asyncio.create_task(_drain(proc.stderr))
    try:
        while True:
            event = json.loads(await socket.receive_text())
            if event.get("type") == "message":
                alive = await _run_turn(socket, proc, event.get("text", ""))
                if not alive:
                    break
    except (WebSocketDisconnect, ValueError):
        pass
    finally:
        drain.cancel()
        await _shutdown(proc)


async def _run_turn(
    socket: WebSocket, proc: asyncio.subprocess.Process, message: str
) -> bool:
    """Run one turn on the persistent process: send, stream events, end with ``done``.

    A stop-watcher reads the socket in parallel so the user can interrupt a long
    autonomous run; on a stop it aborts the turn but leaves the process warm.

    Args:
        socket: The chat WebSocket to push events to.
        proc: The running Claude process.
        message: The user's request for this turn.

    Returns:
        True if the process is still usable for the next turn; False if it died,
        so the caller stops the loop.
    """
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(user_message_line(message))
    await proc.stdin.drain()

    stopped = asyncio.Event()
    watcher = asyncio.create_task(_watch_stop(socket, proc, stopped))
    try:
        saw_result = await _stream_turn(socket, proc, stopped)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

    if stopped.is_set():
        await socket.send_json({"kind": "notice", "text": "stopped"})
    elif not saw_result:  # stdout closed mid-turn: the process is gone
        await socket.send_json({"kind": "error", "message": "the assistant stopped"})
    await socket.send_json({"kind": "done"})
    return saw_result


async def _stream_turn(
    socket: WebSocket, proc: asyncio.subprocess.Process, stopped: asyncio.Event
) -> bool:
    """Stream a turn's stdout to UI events until its result.

    Args:
        socket: The chat WebSocket to push events to.
        proc: The running Claude process.
        stopped: Set when the user asked to stop; suppresses the error event for
            the aborted result so a deliberate stop reads as a stop, not a failure.

    Returns:
        True once a result line is seen, False if stdout closed first.
    """
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if payload.get("type") == "result":
            if not stopped.is_set():
                for ui in to_events(payload):
                    await socket.send_json(ui)
            return True
        for ui in to_events(payload):
            await socket.send_json(ui)
    return False


async def _watch_stop(
    socket: WebSocket, proc: asyncio.subprocess.Process, stopped: asyncio.Event
) -> None:
    """Read the socket during a turn and interrupt the process on a stop.

    The turn's stdout loop does not read the socket, so this reads it in parallel.
    On a ``{"type": "stop"}`` message it records the stop and sends the interrupt
    control request; the in-flight turn then ends with a result.

    Args:
        socket: The chat WebSocket, read for a stop request.
        proc: The running Claude process to interrupt.
        stopped: Set here when a stop arrives, so the turn reports it cleanly.
    """
    try:
        while True:
            event = json.loads(await socket.receive_text())
            if event.get("type") == "stop":
                stopped.set()
                await _interrupt(proc)
    except (WebSocketDisconnect, ValueError):
        pass


async def _interrupt(proc: asyncio.subprocess.Process) -> None:
    """Write the interrupt control request to the process's stdin."""
    if proc.stdin is None:
        return
    proc.stdin.write(interrupt_line())
    await proc.stdin.drain()


async def _drain(stream: asyncio.StreamReader) -> None:
    """Consume a process stream so its pipe never fills and blocks the process."""
    async for _ in stream:
        pass


async def _shutdown(proc: asyncio.subprocess.Process) -> None:
    """Close stdin and wait for the process to exit, terminating if it lingers."""
    if proc.returncode is not None:
        return
    if proc.stdin is not None:
        proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.terminate()
        await proc.wait()
