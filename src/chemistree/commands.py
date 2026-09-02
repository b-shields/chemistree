"""Parse and apply text commands to a design session.

The shared command interpreter over a :class:`DesignSession`. It is surfaced by
both the web app's command box and the MCP tool servers, so it lives in the core
package with no dependency on either.
"""

from __future__ import annotations

from pathlib import Path

from chemistree.session import DesignSession


def run_command(session: DesignSession, text: str) -> str:
    """Apply a text command to the session and return a status message.

    Supported commands::

        swap <id> <group>
        grow <id> <position_id> <group>
        mutate <id> <position_id> <element>
        remove <id>
        minimize <id> [degrees] [window]
        undo
        group <id>
        distance <residue>
        contacts [dist_cutoff]
        residues_near <id> <position_id|-> <cutoff>
        clashes
        write_pose <path>

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
        kept, position = session.remove(int(args[0]))
        return (
            f"removed group {args[0]}; group {kept} now has an open hydrogen at "
            f"position {position} where the group was attached — grow there to "
            f"replace it"
        )
    if command == "minimize":
        degrees = float(args[1]) if len(args) > 1 else 0.0
        window = float(args[2]) if len(args) > 2 else 180.0
        return session.minimize(int(args[0]), degrees, window=window)
    if command == "undo":
        session.undo()
        return "reverted the last edit"
    if command == "group":
        return session.describe_group(int(args[0]))
    if command == "distance":
        return session.distance(args[0])
    if command == "contacts":
        return session.contacts(float(args[0])) if args else session.contacts()
    if command == "residues_near":
        pos = None if args[1] == "-" else int(args[1])
        return session.residues_near(int(args[0]), pos, float(args[2]))
    if command == "clashes":
        return session.clashes()
    if command == "write_pose":
        path = Path(args[0])
        path.write_text(session.pose_sdf())
        return f"wrote pose to {path}"
    raise ValueError(f"unknown command: {command!r}")
