"""Benchmark arms: what tools each agent gets, and how they reach it.

Every arm runs the same model on the same case; only the tool surface differs, so
the comparison isolates the representation. ``chemistree`` gets the MCP tools;
``generalist`` gets Bash (rdkit, and smina for 3D) to do it by hand; ``naked`` gets
no tools and must reason over the SMILES in the prompt.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# Built-in tools blocked so an arm's only capability is what we grant it, mirroring
# the app's chat driver.
_BUILTINS = [
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


@dataclass(frozen=True)
class Arm:
    """One benchmark condition.

    Attributes:
        name: Selector used on the command line and stored on each result row.
        allowed_tools: Value for ``--allowedTools`` (empty to omit the flag).
        disallowed_tools: Value for ``--disallowedTools``.
        uses_chemistree: Whether to attach the chemistree MCP server.
        tools_profile: The server's ``--tools`` profile (``all`` or ``2d``), used
            only when ``uses_chemistree``.
    """

    name: str
    allowed_tools: str
    disallowed_tools: str
    uses_chemistree: bool
    tools_profile: str = "all"


CHEMISTREE = Arm(
    name="chemistree",
    allowed_tools="mcp__chemistree",
    disallowed_tools=",".join(_BUILTINS),
    uses_chemistree=True,
)
GENERALIST = Arm(
    name="generalist",
    allowed_tools="Bash",
    disallowed_tools=",".join(t for t in _BUILTINS if t != "Bash"),
    uses_chemistree=False,
)
NAKED = Arm(
    name="naked",
    allowed_tools="",
    disallowed_tools=",".join([*_BUILTINS, "mcp__chemistree"]),
    uses_chemistree=False,
)

ARMS = {arm.name: arm for arm in (CHEMISTREE, GENERALIST, NAKED)}


def chemistree_mcp_config(profile: str, trace_path: str | None = None) -> dict:
    """The ``--mcp-config`` payload that attaches the standalone chemistree server.

    The server runs on this interpreter, so it resolves in the same environment.

    Args:
        profile: The server's ``--tools`` profile (``all`` or ``2d``).
        trace_path: Optional JSONL path for the server's ``--trace`` mode.

    Returns:
        A dict to serialize as the ``--mcp-config`` JSON file.
    """
    args = ["-m", "chemistree.mcp.server", "--tools", profile]
    if trace_path is not None:
        args += ["--trace", trace_path]
    return {"mcpServers": {"chemistree": {"command": sys.executable, "args": args}}}
