"""Parse and apply text commands to a design session (the app's command box)."""

from __future__ import annotations

from chemistree.session import DesignSession


def run_command(session: DesignSession, text: str) -> str:
    """Apply a text command to the session and return a status message.

    Supported commands::

        swap <id> <group>
        grow <id> <position_id> <group>
        mutate <id> <position_id> <element>
        remove <id>
        undo
        group <id>
        distance <residue>

    Ids come from ``describe`` (the group overview); atom position ids come from
    ``group <id>`` (``describe_group``).

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
        return f"swapped group {args[0]}"
    if command == "grow":
        node_id, position_id, group = args[0], args[1], " ".join(args[2:])
        session.grow(int(node_id), int(position_id), group)
        return f"grew {group} at position {position_id}"
    if command == "mutate":
        node_id, position_id, element = args[0], args[1], args[2]
        session.mutate(int(node_id), int(position_id), element)
        return f"mutated position {position_id} to {element}"
    if command == "remove":
        session.remove(int(args[0]))
        return f"removed group {args[0]}"
    if command == "undo":
        session.undo()
        return "reverted the last edit"
    if command == "group":
        return session.describe_group(int(args[0]))
    if command == "distance":
        return session.distance(args[0])
    raise ValueError(f"unknown command: {command!r}")
