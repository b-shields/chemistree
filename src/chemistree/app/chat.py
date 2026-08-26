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
import json
import logging
import os
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

_log = logging.getLogger(__name__)

# The Claude binary and model. Override the binary for a non-default install.
_CLAUDE_BIN = os.environ.get("CHEMISTREE_CLAUDE_BIN", "claude")
_MODEL = "haiku"
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

    Args:
        tools: Comma-separated tool names the mode exposes, named in the prompt.
        primed_note: Extra guidance appended for the primed mode.

    Returns:
        The full ``--append-system-prompt`` text.
    """
    return (
        f"You are a medicinal chemist talking through a structure with a colleague "
        f"at the bench. You edit one molecule in a live design session, using only "
        f"the chemistree tools ({tools}) to inspect and change it. A group can be a "
        f"common name (isopropyl) or a SMILES with one dummy [*] per attachment "
        f"point ([*]C1([*])COC1 for a 2-port oxetane linker); if a name is not "
        f"recognized, pass a SMILES. Group ids come from the group listing; before "
        f"grow or mutate, call describe_group(id) to get the atom position ids. "
        f"When a request names a residue (near/closest to it), call distance to see "
        f"which group is closest, or contacts to map the whole binding site, before "
        f"editing. After adding or changing a group, call clashes to check the new "
        f"pose; if a group clashes with another group or the protein, tell the user "
        f"and offer to rotate it to relieve the clash, and rotate only once they "
        f"agree (rotate turns a group about its attachment bond, carrying its "
        f"substituents, and settles it to the least-clashing angle near what you "
        f"ask).\n\n"
        f"The ids, atom positions, and tables the tools return are your private "
        f"scaffolding for addressing atoms — never repeat them to the user. Talk the "
        f"way a chemist talks: name each group by what it is (the dichlorophenyl, "
        f"the pyrimidine core, the para hydroxyl), describe positions as "
        f"ortho/meta/para or by the atoms involved, and write in flowing sentences, "
        f"not bracketed ids, atom numbers, or copied tables. Read the SMILES and "
        f"atom map to recognize the real chemistry rather than leaning on the tools' "
        f"generic labels. Save headers and bullet lists for when the user asks for a "
        f"breakdown; otherwise reply in a short, natural paragraph — one sentence to "
        f"confirm an edit, a few plain sentences when asked to explain. Never read, "
        f"write, or run files or shell commands.{primed_note}"
    )


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
        " You are given the current group listing, refreshed after every edit; use "
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


def build_command(mode: ChatMode, context: str = "") -> list[str]:
    """Build the ``claude`` argv for the persistent streaming session.

    The process is spawned once and reused for every turn. It reads user
    messages from stdin as stream-json and writes stream-json to stdout, so no
    per-turn message or ``--resume`` id is needed here.

    Args:
        mode: The chat mode, which chooses the system prompt.
        context: The current fragment listing to seed. Used only when the mode
            primes state; ignored otherwise.

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
        _MODEL,
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
    socket: WebSocket, mode: ChatMode = DEFAULT_MODE, context: str = ""
) -> None:
    """Bridge a browser chat panel to one persistent Claude Code process.

    Spawns the process once, then feeds each inbound ``{"type": "message"}`` to
    its stdin and streams the reply back until the turn ends. The process is
    reused across turns and torn down when the socket closes.

    Args:
        socket: The accepted WebSocket.
        mode: The chat mode, which chooses the system prompt and whether to prime.
        context: The current fragment listing to seed when the mode primes.
    """
    await socket.accept()
    # In the primed mode the MCP server must attach the listing to edit results;
    # it is a child of this process, so pass the flag down through the environment.
    env = {**os.environ, _PRIME_ENV: "1"} if mode.prime_context else None
    proc = await asyncio.create_subprocess_exec(
        *build_command(mode, context),
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

    saw_result = False
    async for raw in proc.stdout:
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        for ui in to_events(payload):
            await socket.send_json(ui)
        if payload.get("type") == "result":
            saw_result = True
            break

    if not saw_result:  # stdout closed mid-turn: the process is gone
        await socket.send_json({"kind": "error", "message": "the assistant stopped"})
    await socket.send_json({"kind": "done"})
    return saw_result


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
