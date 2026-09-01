"""Agent guidance, split so it can be delivered fairly across benchmark arms.

The demo's ``app/system_prompt.md`` stays the tuned, voiced prompt for the web app. The
same guidance lives here as two markdown files:

- ``medchem.md`` — the tool-agnostic medicinal-chemistry playbook (role, drug-likeness,
  ionizable groups, the tiered optimize strategy, the edit budget). Every benchmark arm
  gets it, so domain knowledge is matched.
- ``tools.md`` — how to use the chemistree tools (the score and profile lines, the
  pocket and clash tools, core-hop mechanics). It travels with the chemistree arm only,
  as the representation's own API.

The chemistree arm gets medchem + tools; a no-tools or generalist arm gets medchem.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS = Path(__file__).parent / "prompts"
_MEDCHEM = (_PROMPTS / "medchem.md").read_text().strip()
_TOOLS = (_PROMPTS / "tools.md").read_text().strip()

# The full chemistree tool set, named in the tools guidance.
_TOOL_NAMES = (
    "describe, describe_group, smiles, swap, grow, mutate, remove, minimize, undo, "
    "distance, contacts, clashes, write_pose"
)


def medchem_guidance() -> str:
    """The shared, tool-agnostic medicinal-chemistry playbook (all arms)."""
    return _MEDCHEM


def chemistree_guidance() -> str:
    """The medchem playbook plus chemistree tool-usage (chemistree arm and server)."""
    return f"{_MEDCHEM}\n\n{_TOOLS.format(tools=_TOOL_NAMES)}"
