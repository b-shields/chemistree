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
# Set by the chat driver (app/chat.py, _PRIME_ENV = "CHEMISTREE_PRIME") when the
# primed mode is on. Then edit results carry the refreshed fragment listing, so
# the agent acts on node ids without a separate find/describe call.
_PRIME = os.environ.get("CHEMISTREE_PRIME") == "1"
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


@mcp.tool
def describe() -> str:
    """List the molecule's groups with their ids, names, and connections."""
    return str(_state()["describe"])


@mcp.tool
def describe_group(group_id: int) -> str:
    """Show one group's atom positions, rings, and neighbourhood.

    Call this before ``grow`` or ``mutate`` to get the atom **position ids**: each
    heavy atom, the ids of its hydrogens (grow targets), and its neighbours as
    ortho/meta/para or greek terms relative to the group's substituents.

    Args:
        group_id: Id of the group to detail (from ``describe``).
    """
    return _command(f"group {group_id}")


@mcp.tool
def smiles() -> str:
    """Canonical SMILES of the current molecule."""
    return str(_state()["smiles"])


@mcp.tool
def swap(node_id: int, group: str) -> str:
    """Replace the whole group at ``node_id`` with a new group.

    The group is a common name ('trifluoromethyl') or a SMILES with one dummy
    ``[*]`` per attachment point ('[*]C1([*])COC1' for a 2-port oxetane linker);
    if a name is not recognized, pass a SMILES.

    Args:
        node_id: Group whose fragment is replaced (from ``describe``).
        group: A common group name or a SMILES with a ``[*]`` per port.
    """
    return _command(f"swap {node_id} {group}", with_state=_PRIME)


@mcp.tool
def grow(node_id: int, position_id: int, group: str) -> str:
    """Grow a group where a hydrogen is, at a specific position.

    Get ``position_id`` from ``describe_group(node_id)`` — it is the id of a
    hydrogen on the atom you want to grow from (e.g. the hydrogen ortho to a named
    substituent).

    Args:
        node_id: Group bearing the hydrogen (from ``describe``).
        position_id: Id of the hydrogen to replace (from ``describe_group``).
        group: A common group name, or a SMILES with one dummy ``[*]`` port.
    """
    return _command(f"grow {node_id} {position_id} {group}", with_state=_PRIME)


@mcp.tool
def mutate(node_id: int, position_id: int, element: str) -> str:
    """Change one heavy atom's element (e.g. a ring carbon to N for a pyridine).

    Get ``position_id`` from ``describe_group(node_id)`` — it is the id of the
    heavy atom to change (e.g. the ring carbon meta to a named substituent).

    Args:
        node_id: Group to edit (from ``describe``).
        position_id: Id of the heavy atom to change (from ``describe_group``).
        element: New element as a symbol ("N") or name ("nitrogen").
    """
    return _command(f"mutate {node_id} {position_id} {element}", with_state=_PRIME)


@mcp.tool
def remove(node_id: int) -> str:
    """Delete a leaf group, capping its parent with hydrogen.

    Use this to prune a terminal group or ring, e.g. "delete the phenol ring".
    Only a leaf can be removed; an internal linker raises an error.

    Args:
        node_id: Leaf group to remove (from ``describe``).
    """
    return _command(f"remove {node_id}", with_state=_PRIME)


@mcp.tool
def undo() -> str:
    """Revert the most recent edit (swap, grow, mutate, or remove)."""
    return _command("undo", with_state=_PRIME)


@mcp.tool
def distance(residue: str) -> str:
    """Report how close each group is to a receptor residue.

    Use this when a request names a residue (e.g. "near ASP") to see which group is
    closest before editing: it returns a table of per-group minimum distances,
    closest first, then per-atom detail.

    Args:
        residue: Residue name ('ASP') or name with number ('ASP381').
    """
    return _command(f"distance {residue}")


@mcp.tool
def contacts(dist_cutoff: float = 4.5) -> str:
    """Map the binding site: the closest group and atom for each nearby residue.

    Use this to see how the molecule sits in the pocket, or before targeting a
    residue. It returns one row per residue within ``dist_cutoff`` of any ligand
    heavy atom, naming the closest group and the atom in it — so a request like
    "grow toward the aspartate" maps straight to a group and atom to edit.

    Args:
        dist_cutoff: Site radius in angstrom (a residue counts when any atom is
            within this distance of any ligand heavy atom).
    """
    return _command(f"contacts {dist_cutoff}")


if __name__ == "__main__":
    mcp.run()
