"""Parse and apply text commands to a design session (the app's command box)."""

from __future__ import annotations

from chemistree.session import DesignSession


def run_command(session: DesignSession, text: str) -> str:
    """Apply a text command to the session and return a status message.

    Supported commands::

        swap <id> <group>
        add <id> <group> <position> <reference>
        undo <id>
        find <name>
        nearest <name> <residue>

    Args:
        session: The session to edit.
        text: The command line.

    Returns:
        A human-readable status message.

    Raises:
        ValueError: If the command is unknown or malformed.
    """
    parts = text.split()
    if not parts:
        return ""
    command, args = parts[0], parts[1:]

    if command == "swap":
        session.swap(int(args[0]), " ".join(args[1:]))
        return f"swapped node {args[0]}"
    if command == "add":
        node_id, group, position, reference = args[0], args[1], args[2], args[3]
        session.add(
            int(node_id), group, position=_position(position), reference=reference
        )
        return f"grew {group} on node {node_id}"
    if command == "undo":
        session.undo(int(args[0]))
        return f"reverted node {args[0]}"
    if command == "find":
        return f"{args[0]}: {session.find(name=args[0])}"
    if command == "nearest":
        name, residue = args[0], args[1]
        return f"{name} nearest {residue}: node {session.nearest(name, residue)}"
    raise ValueError(f"unknown command: {command!r}")


def _position(text: str) -> int | str:
    """A bond count if numeric, else a ring synonym (ortho/meta/para)."""
    return int(text) if text.lstrip("-").isdigit() else text
