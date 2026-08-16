"""Agent-facing description of a fragment tree, in two tiers.

``describe_tree`` renders a compact overview: one line per group with its id,
name, fragment SMILES, formula, role, and inter-group connections. ``describe_group``
renders the heavy per-group detail on demand — an **Atom Map** (atom-mapped SMILES
whose map numbers are the exact rdkit atom ids), a **Rings** section (natural-language
ring membership), and a **Topology** matrix (pairwise bond distances over the
heavy-atom + port skeleton). Splitting them keeps a large molecule from flooding the
context: the agent orients on the overview, then pulls detail for the one group it
edits.

Two id namespaces are kept visually distinct: ``:k`` is an atom id (a grow/mutate
position), ``[n*]`` is a port (a connection between groups).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rdkit import Chem

from chemistree.annotations import NodeAnnotation, annotate

if TYPE_CHECKING:
    from chemistree.tree import FragmentTree

Labels = dict[int, NodeAnnotation]

_OVERVIEW_NOTE = (
    "Call describe_group(id) for a group's atom positions, rings, and topology."
)

_GROUP_LEGEND = (
    "Atom ids are `:k` (use as grow/mutate position_id); `[n*]` = port "
    "(connects groups).\n"
    "Topology cell = bonds between the row and column atom; diagonal is `.`."
)

_MAP_ID = re.compile(r":(\d+)]")


def describe_tree(tree: FragmentTree) -> str:
    """Render the compact group inventory: one line per group.

    This is the overview the agent reads to pick a group. It lists each group's
    id, name, fragment SMILES, formula, role, and inter-group connections, but not
    the per-group atom positions, rings, or topology — those are pulled on demand
    with :func:`describe_group`, so a large molecule does not flood the context.

    Args:
        tree: The tree to describe.

    Returns:
        A markdown overview of every group.
    """
    nodes = annotate(tree).nodes
    labels = {node.id: node for node in nodes}
    lines = [f"# Group Summary\n{_OVERVIEW_NOTE}\n"]
    lines += [f"- {_header_line(node, labels)}" for node in nodes]
    return "\n".join(lines)


def describe_group(tree: FragmentTree, group_id: int) -> str:
    """Render one group's atom positions, rings, and topology.

    Args:
        tree: The tree the group belongs to.
        group_id: Id of the group to detail.

    Returns:
        A markdown section: the group header and legend, its Atom Map, its Rings
        (when cyclic), and its Topology matrix.
    """
    labels = {node.id: node for node in annotate(tree).nodes}
    node = labels[group_id]
    mol = tree.node(group_id).current.mol
    body = [
        f"## {_header_line(node, labels)}",
        _GROUP_LEGEND,
        "",
        f"**Atom Map:** `{atom_map(mol)}`",
    ]
    rings = rings_section(mol)
    if rings:
        body.append(rings)
    body.append(f"**Topology:**\n```\n{topology_matrix(mol)}\n```")
    return "\n".join(body)


def _header_line(node: NodeAnnotation, labels: Labels) -> str:
    """The one-line identity of a group, reused as its ``##`` header."""
    line = f"[{node.id}] {_label(node)} `{node.smiles}` ({node.formula}) — {node.role}"
    if node.neighbors:
        attached = ", ".join(
            f"[{ref.node_id}] {_label(labels[ref.node_id])}" for ref in node.neighbors
        )
        line += f", attached to {attached}"
    return line


def _label(node: NodeAnnotation) -> str:
    """A group's display label: its common name, or its coarse class."""
    return node.name or node.classification


def atom_map(mol: Chem.Mol) -> str:
    """Atom-mapped SMILES whose map numbers are the exact rdkit atom ids.

    Heavy atoms and hydrogens carry ``:{idx}``; ports render as ``[n*]`` (their
    pairing label), never as an atom id. RDKit cannot emit the atom-map number 0,
    so the map is written as ``idx + 1`` and every ``:k`` is then decremented,
    giving a clean ``:0`` for the first atom.

    Args:
        mol: The fragment molecule (with explicit hydrogens and dummy ports).

    Returns:
        The atom-mapped SMILES string.
    """
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    smiles = Chem.MolToSmiles(work, canonical=False)
    return _MAP_ID.sub(lambda m: f":{int(m.group(1)) - 1}]", smiles)


def ring_ids(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """SSSR rings as tuples of heavy-atom ids, in a stable order.

    Args:
        mol: The fragment molecule.

    Returns:
        One tuple of sorted atom ids per ring, rings ordered by their atoms.
    """
    rings = [tuple(sorted(ring)) for ring in mol.GetRingInfo().AtomRings()]
    return sorted(rings)


def rings_section(mol: Chem.Mol) -> str:
    """The ``Rings:`` bullets naming each ring's atoms, or ``""`` when acyclic.

    Args:
        mol: The fragment molecule.

    Returns:
        A markdown ``**Rings:**`` block, one ``Ring A/B/…`` bullet per SSSR ring,
        or an empty string when the fragment has no ring.
    """
    rings = ring_ids(mol)
    if not rings:
        return ""
    lines = ["**Rings:**"]
    for name, ring in zip(_ring_names(len(rings)), rings):
        atoms = ", ".join(str(i) for i in ring)
        lines.append(f"- Ring {name} ({len(ring)}-membered): atoms {atoms}")
    return "\n".join(lines)


def _ring_names(count: int) -> list[str]:
    """Ring labels A, B, C, … for ``count`` rings."""
    return [chr(ord("A") + i) for i in range(count)]


def topology_matrix(mol: Chem.Mol) -> str:
    """A monospace distance matrix over the heavy-atom + port skeleton.

    Rows and columns are the heavy atoms (labeled by id) and ports (labeled
    ``[n*]``), ordered by rdkit index. Each cell is the topological distance in
    bonds between its row and column atom; the diagonal is ``.``. Hydrogens are
    excluded — an H's distances are its parent's plus one.

    Args:
        mol: The fragment molecule.

    Returns:
        The rendered matrix (no trailing newline).
    """
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    labels = [_atom_label(a) for a in atoms]
    dmat = Chem.GetDistanceMatrix(mol)

    def cell(i: int, j: int) -> str:
        """The matrix entry for two skeleton atoms."""
        return "." if i == j else str(int(dmat[atoms[i].GetIdx()][atoms[j].GetIdx()]))

    width = max(len(label) for label in labels)
    width = max(width, 1) + 1
    head = " " * width + "".join(label.rjust(width) for label in labels)
    lines = [head]
    for i, label in enumerate(labels):
        row = label.rjust(width) + "".join(
            cell(i, j).rjust(width) for j in range(len(atoms))
        )
        lines.append(row)
    return "\n".join(lines)


def _atom_label(atom: Chem.Atom) -> str:
    """A skeleton atom's column label: ``[n*]`` for a port, else its id."""
    if atom.GetAtomicNum() == 0:
        return f"[{atom.GetIsotope()}*]"
    return str(atom.GetIdx())
