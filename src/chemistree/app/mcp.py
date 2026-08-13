"""MCP server exposing the running chemistree app to Claude Code.

Tools are thin clients of the web app (``chemistree.app``): each one calls the
running server over HTTP, so edits made by the agent go through the same session
and update the live 2D/3D viewers. Start the app first, then run this over stdio
(Claude Code launches it via ``.mcp.json``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from fastmcp import FastMCP

BASE_URL = os.environ.get("CHEMISTREE_URL", "http://127.0.0.1:8000")
mcp = FastMCP("chemistree")


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


def _command(text: str) -> str:
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

    Always use this to pick the node when a request mentions a residue (e.g. "the
    chloro near ALA37"): it measures distance, so it resolves *which* matching
    fragment the request means. Do not guess the node in that case.

    Args:
        name: Fragment common name (e.g. 'chloro').
        residue: Residue name in the receptor (e.g. 'PHE').
    """
    return _command(f"nearest {name} {residue}")


@mcp.tool
def swap(node_id: int, group: str) -> str:
    """Replace the fragment at ``node_id`` with a group.

    The group is a common name ('trifluoromethyl') or a SMILES with one dummy
    ``[*]`` per attachment point ('[*]C1([*])COC1' for a 2-port oxetane linker);
    if a name is not recognized, pass a SMILES. When the request names a residue,
    get ``node_id`` from ``nearest`` first — do not guess which fragment is meant.

    Args:
        node_id: Node whose fragment is replaced.
        group: A common group name or a SMILES with a ``[*]`` per port.
    """
    return _command(f"swap {node_id} {group}")


@mcp.tool
def add(node_id: int, group: str, position: str, reference: str) -> str:
    """Grow a group on a scaffold, relative to one of its substituents.

    When the request names a residue, get ``node_id`` from ``nearest`` first.

    Args:
        node_id: Scaffold node to grow from.
        group: A common group name, or a SMILES with one dummy ``[*]`` port.
        position: A bond count, or a synonym (ortho/meta/para, alpha/beta/gamma).
        reference: Name of the scaffold substituent to count from.
    """
    return _command(f"add {node_id} {group} {position} {reference}")


@mcp.tool
def mutate(
    node_id: int,
    element: str,
    between_first: str = "",
    between_second: str = "",
    position: str = "",
    reference: str = "",
) -> str:
    """Change one ring atom's element (e.g. aromatic C to N for a pyridine).

    Address the atom either by the two substituents it sits between, or by a
    position relative to one substituent. Use this for ring heteroatom edits,
    e.g. an aniline to a 2-aminopyridine: mutate the ring carbon between the
    amino and the methyl to nitrogen.

    Args:
        node_id: Ring node to edit.
        element: New element as a symbol ("N") or name ("nitrogen").
        between_first: One substituent the target atom sits between.
        between_second: The other substituent it sits between.
        position: A bond count or ring synonym from ``reference`` (alternative
            to ``between``).
        reference: Name of the substituent to count from (with ``position``).
    """
    if between_first and between_second:
        return _command(
            f"mutate {node_id} {element} between {between_first} {between_second}"
        )
    return _command(f"mutate {node_id} {element} {position} {reference}")


@mcp.tool
def remove(node_id: int) -> str:
    """Delete a leaf fragment, capping its parent with hydrogen.

    Use this to prune a terminal group or ring, e.g. "delete the phenol ring".
    Only a leaf can be removed; an internal linker raises an error.

    Args:
        node_id: Leaf node to remove.
    """
    return _command(f"remove {node_id}")


@mcp.tool
def undo(node_id: int) -> str:
    """Revert the most recent edit at ``node_id``."""
    return _command(f"undo {node_id}")


if __name__ == "__main__":
    mcp.run()
