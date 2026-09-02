"""The chemistree MCP tool set, registered against a backend.

One source of the tools and their docstrings — the text the agent reads to
understand chemistree. Two backends register the same set: the in-process backend
in ``server.py`` (the standalone ``chemistree-mcp``) and the HTTP backend in
``app/mcp.py`` (the web app, so its viewers update). A backend runs a text command
and reports the session's current state; ``prime`` says whether edit results carry
the refreshed group listing so the agent skips a lookup.
"""

from __future__ import annotations

from typing import Protocol

from fastmcp import FastMCP


class Backend(Protocol):
    """What a tool needs from its host: run a command, read state, know the mode."""

    prime: bool

    def run(self, text: str, *, with_state: bool = False) -> str:
        """Run a text command and return its result message (and SMILES)."""
        ...

    def state(self) -> dict:
        """The current viewer state, with ``smiles`` and ``describe`` keys."""
        ...


def register(mcp: FastMCP, backend: Backend, *, profile: str = "all") -> None:
    """Register the chemistree tool set on ``mcp``, driven by ``backend``.

    Args:
        mcp: The FastMCP server to add tools to.
        backend: The host that runs commands and reports state.
        profile: ``"all"`` registers every tool; ``"2d"`` registers only the 2D
            editing set (no pocket or pose tools), so a receptor-free task is not
            given tools it cannot use.
    """

    @mcp.tool
    def describe() -> str:
        """List the molecule's groups with their ids, names, and connections."""
        return str(backend.state()["describe"])

    @mcp.tool
    def describe_group(group_id: int) -> str:
        """Show one group's atom positions, rings, and neighbourhood.

        Call this before ``grow`` or ``mutate`` to get the atom **position ids**:
        each heavy atom, the ids of its hydrogens (grow targets), and its
        neighbours by bond distance — ortho/meta/para on a plain benzene ring,
        otherwise a plain bond count (fused rings included) — each shown as an
        ``[element:id]`` token.

        Args:
            group_id: Id of the group to detail (from ``describe``).
        """
        return backend.run(f"group {group_id}")

    @mcp.tool
    def smiles() -> str:
        """Canonical SMILES of the current molecule."""
        return str(backend.state()["smiles"])

    @mcp.tool
    def swap(node_id: int, group: str) -> str:
        """Replace the whole group at ``node_id`` with a new group.

        The group is a common name ('trifluoromethyl') or a SMILES with one dummy
        ``[*]`` per attachment point ('[*]C1([*])COC1' for a 2-port oxetane
        linker); if a name is not recognized, pass a SMILES.

        The new group must have the same number of ports as the group it replaces.
        Use the group's own port labels (from ``describe_group``) so each
        attachment keeps its place — bare ``[*]`` dummies are assigned in atom
        order and can put substituents in the wrong spots. To change a group and
        drop a substituent, ``remove`` that substituent leaf first (freeing its
        port), then swap the lower-port group. A port-count error names the ports
        and how to proceed.

        When users ask to add a heterocycle they will typically use canonical
        numbering (atomic number priority around the ring) to refer to the H
        position(s) that should carry a port. Examples: (A) "add a 2-oxazole"
        means `c1cnc([*])o1`. (B) Suppose there is a benzene with 2 ports
        (`c1([1*])cc([2*])ccc1`) where `[1*]F` and `[2*]C` are attached groups;
        then if a user says "swap the F, Me phenyl to a 2,5 thiazole" it means
        `c1([2*])cnc([1*])s1`.

        Args:
            node_id: Group whose fragment is replaced (from ``describe``).
            group: A common group name or a SMILES with a ``[*]`` per port.
        """
        return backend.run(f"swap {node_id} {group}", with_state=backend.prime)

    @mcp.tool
    def grow(node_id: int, position_id: int, group: str) -> str:
        """Grow a group where a hydrogen is, at a specific position.

        Get ``position_id`` from ``describe_group(node_id)`` — it is the id of a
        hydrogen on the atom you want to grow from (e.g. the hydrogen ortho to a
        named substituent).

        Args:
            node_id: Group bearing the hydrogen (from ``describe``).
            position_id: Id of the hydrogen to replace (from ``describe_group``).
            group: A common group name, or a SMILES with one dummy ``[*]`` port.
        """
        return backend.run(
            f"grow {node_id} {position_id} {group}", with_state=backend.prime
        )

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
        return backend.run(
            f"mutate {node_id} {position_id} {element}", with_state=backend.prime
        )

    @mcp.tool
    def remove(node_id: int) -> str:
        """Delete a leaf group, capping its parent with hydrogen.

        Use this to prune a terminal group or ring, e.g. "delete the phenol ring".
        Only a leaf can be removed; an internal linker raises an error. The result
        names the open hydrogen position left where the group was attached, so you
        can grow a replacement at that exact spot.

        Args:
            node_id: Leaf group to remove (from ``describe``).
        """
        return backend.run(f"remove {node_id}", with_state=backend.prime)

    @mcp.tool
    def undo() -> str:
        """Revert the most recent edit (swap, grow, mutate, or remove)."""
        return backend.run("undo", with_state=backend.prime)

    if profile == "2d":  # no pocket or pose tools for a receptor-free task
        return

    @mcp.tool
    def distance(residue: str) -> str:
        """Report how close each group is to a receptor residue.

        Use this when a request names a residue (e.g. "near ASP") to see which
        group is closest before editing: it returns a table of per-group minimum
        distances, closest first, then per-atom detail.

        Args:
            residue: Residue name ('ASP') or name with number ('ASP381').
        """
        return backend.run(f"distance {residue}")

    @mcp.tool
    def contacts(dist_cutoff: float = 4.5) -> str:
        """Map the binding site: the closest group and atom for each nearby residue.

        Use this to see how the molecule sits in the pocket, or before targeting a
        residue. It returns one row per residue within ``dist_cutoff`` of any
        ligand heavy atom, naming the closest group and the atom in it — so a
        request like "grow toward the aspartate" maps straight to a group and atom
        to edit.

        Args:
            dist_cutoff: Site radius in angstrom (a residue counts when any atom is
                within this distance of any ligand heavy atom).
        """
        return backend.run(f"contacts {dist_cutoff}")

    @mcp.tool
    def minimize(group_id: int, degrees: float = 0.0, window: float = 180.0) -> str:
        """Settle a group about its attachment bond into its best-scoring rotamer.

        The group turns about the single bond joining it to the rest of the
        molecule, carrying its own substituents. Every whole-degree turn is scored
        by the full Vinardo energy — the ligand's own internal strain plus, with a
        receptor, the protein-ligand fit — and the best turn is applied. Use this
        to settle a group after an edit that added or moved more than one heavy
        atom (call ``clashes`` to find a strained group). The result reports the
        applied turn and the energy before and after.

        Args:
            group_id: Group to settle (from ``describe``).
            degrees: Requested turn, in degrees; 0 settles from the current pose
                over the whole circle.
            window: How far, in degrees, the search may stray from ``degrees``.
        """
        return backend.run(
            f"minimize {group_id} {degrees} {window}", with_state=backend.prime
        )

    @mcp.tool
    def clashes() -> str:
        """List steric clashes in the current pose, worst first.

        Reports group-group overlaps and, when a receptor is loaded, each group
        that overlaps a residue. Call this after an edit to check whether a new or
        moved group clashes; if it does, offer to ``minimize`` that group to
        relieve it.
        """
        return backend.run("clashes")

    @mcp.tool
    def write_pose(path: str) -> str:
        """Save the current 3D pose to an SDF file.

        Writes the molecule with its conformer and explicit hydrogens as one SDF
        record. Needs a ligand with 3D coordinates.

        Args:
            path: File path to write the SDF to.
        """
        return backend.run(f"write_pose {path}")
