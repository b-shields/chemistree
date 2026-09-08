"""MCP server exposing the running chemistree app to Claude Code.

Thin HTTP clients of the web app (``chemistree.app``): each tool calls the running
server over HTTP, so edits made by the agent go through the same session and update
the live 2D/3D viewers. The tool set and its docstrings live in
``chemistree.mcp.tools``; this module supplies the HTTP backend and keeps the
error/result helpers. Start the app first, then run this over stdio (Claude Code
launches it via ``.mcp.json``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from fastmcp import FastMCP

from chemistree.mcp import tools

BASE_URL = os.environ.get("CHEMISTREE_URL", "http://127.0.0.1:8000")
# Set by the chat driver (app/chat.py, _PRIME_ENV = "CHEMISTREE_PRIME") when the
# primed mode is on. Then edit results carry the refreshed fragment listing, so
# the agent acts on node ids without a separate find/describe call.
_PRIME = os.environ.get("CHEMISTREE_PRIME") == "1"


def _state() -> dict:
    """The app's current viewer state."""
    with urllib.request.urlopen(f"{BASE_URL}/state") as response:
        return dict(json.load(response))


def read_error(http_error: urllib.error.HTTPError) -> str:
    """Return the app's error text from a failed request body.

    Args:
        http_error: The error ``urlopen`` raises on a 4xx/5xx response.

    Returns:
        The ``error`` field from the JSON body when present, else the raw body,
        else the status line. This is what lets the agent see *why* an edit
        failed instead of a bare "HTTP Error 400".
    """
    body = http_error.read().decode(errors="ignore")
    try:
        payload = json.loads(body)
    except ValueError:
        return body.strip() or str(http_error)
    return str(payload.get("error", body))


def format_result(data: dict, *, with_state: bool) -> str:
    """Format a ``/command`` response for the agent.

    Args:
        data: The decoded ``/command`` JSON, with ``message`` and ``state``.
        with_state: Append the current fragment listing (node ids, names,
            connections) after the SMILES, so the agent can act on the new state
            without a separate find/describe call.

    Returns:
        A one-line result, plus the fragment listing when ``with_state``.
    """
    base = f"{data['message']} | SMILES: {data['state']['smiles']}"
    if with_state:
        return f"{base}\n\nCurrent fragments:\n{data['state']['describe']}"
    return base


def _command(text: str, *, with_state: bool = False) -> str:
    """Run a command against the app and report the result and new SMILES."""
    request = urllib.request.Request(
        f"{BASE_URL}/command",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
    except urllib.error.HTTPError as http_error:
        return f"error: {read_error(http_error)}"
    if "error" in data:
        return f"error: {data['error']}"
    return format_result(data, with_state=with_state)


class HttpBackend:
    """A tools backend that drives the running web app over HTTP."""

    prime = _PRIME

    def run(self, text: str, *, with_state: bool = False) -> str:
        """Run a command against the app; see :func:`_command`."""
        return _command(text, with_state=with_state)

    def state(self) -> dict:
        """The app's current viewer state; see :func:`_state`."""
        return _state()


mcp = FastMCP("chemistree")
tools.register(mcp, HttpBackend())


if __name__ == "__main__":
    mcp.run()
