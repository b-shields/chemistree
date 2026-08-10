"""MCP server exposing the running chemistree app to Claude Code.

Tools are thin clients of the web app (``chemistree.app``): each one calls the
running server over HTTP, so edits made by the agent go through the same session
and update the live 2D/3D viewers. Start the app first, then run this over stdio
(Claude Code launches it via ``.mcp.json``).
"""

from __future__ import annotations

import json
import os
import urllib.request

from fastmcp import FastMCP

BASE_URL = os.environ.get("CHEMISTREE_URL", "http://127.0.0.1:8000")
mcp = FastMCP("chemistree")


def _state() -> dict:
    """The app's current viewer state."""
    with urllib.request.urlopen(f"{BASE_URL}/state") as response:
        return dict(json.load(response))


def _command(text: str) -> str:
    """Run a command against the app and report the result and new SMILES."""
    request = urllib.request.Request(
        f"{BASE_URL}/command",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        data = json.load(response)
    if "error" in data:
        return f"error: {data['error']}"
    return f"{data['message']} | SMILES: {data['state']['smiles']}"


@mcp.tool
def describe() -> str:
    """List the molecule's fragments with their node ids, names, and connections."""
    return str(_state()["describe"])


@mcp.tool
def smiles() -> str:
    """Canonical SMILES of the current molecule."""
    return str(_state()["smiles"])


@mcp.tool
def find(name: str) -> str:
    """Node ids of fragments with a common name (e.g. 'chloro', 'phenyl')."""
    return _command(f"find {name}")


@mcp.tool
def nearest(name: str, residue: str) -> str:
    """Node id of the fragment named ``name`` closest to a receptor residue.

    Args:
        name: Fragment common name (e.g. 'chloro').
        residue: Residue name in the receptor (e.g. 'PHE').
    """
    return _command(f"nearest {name} {residue}")


@mcp.tool
def swap(node_id: int, group: str) -> str:
    """Replace the fragment at ``node_id`` with a group (a common name or SMILES)."""
    return _command(f"swap {node_id} {group}")


@mcp.tool
def add(node_id: int, group: str, position: str, reference: str) -> str:
    """Grow a group on a scaffold, relative to one of its substituents.

    Args:
        node_id: Scaffold node to grow from.
        group: Group to add (a common name or SMILES).
        position: A bond count, or a ring synonym (ortho/meta/para).
        reference: Name of the scaffold substituent to count from.
    """
    return _command(f"add {node_id} {group} {position} {reference}")


@mcp.tool
def undo(node_id: int) -> str:
    """Revert the most recent edit at ``node_id``."""
    return _command(f"undo {node_id}")


if __name__ == "__main__":
    mcp.run()
