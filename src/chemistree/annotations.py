"""Agent-facing description of a fragment tree.

One typed data model with two serializations: ``to_dict`` (canonical JSON for
agents / MCP tools, with stable node ids) and ``to_markdown`` (a readable menu for
humans and the demo terminal). Data is kept separate from presentation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Optional

from rdkit import Chem

from chemistree.fragment import Fragment
from chemistree.naming import classify_fragment, name_fragment

if TYPE_CHECKING:
    from chemistree.tree import FragmentNode, FragmentTree

Namer = Callable[[Fragment], Optional[str]]


@dataclass(frozen=True)
class AtomAnnotation:
    """A single addressable atom, for modify/grow requests."""

    index: int
    element: str
    aromatic: bool
    in_ring: bool
    num_h: int  # attached hydrogens = available growth sites


@dataclass(frozen=True)
class NeighborRef:
    """A link to an adjacent node, with the port connecting them."""

    node_id: int
    port: int
    name: str | None


@dataclass(frozen=True)
class NodeAnnotation:
    """Everything an agent needs to recognize and address one fragment."""

    id: int
    smiles: str
    formula: str
    classification: str
    role: str  # "leaf" | "scaffold"
    is_ring: bool
    name: str | None
    ports: list[int]
    neighbors: list[NeighborRef]
    atoms: list[AtomAnnotation] | None


@dataclass(frozen=True)
class TreeAnnotation:
    """A description of every node in a fragment tree."""

    nodes: list[NodeAnnotation]

    def to_dict(self) -> dict:
        """Canonical JSON-serializable form."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Readable menu, one line per node (plus atoms when present)."""
        label_of = {n.id: (n.name or n.classification) for n in self.nodes}
        lines = []
        for node in self.nodes:
            line = (
                f"- **[{node.id}] {label_of[node.id]}** `{node.smiles}` "
                f"({node.formula}) — {node.role}"
            )
            if node.neighbors:
                attached = ", ".join(
                    f"[{n.node_id}] {label_of[n.node_id]}" for n in node.neighbors
                )
                line += f", attached to {attached}"
            lines.append(line)
            for atom in node.atoms or []:
                lines.append(
                    f"    - atom {atom.index}: {atom.element}"
                    f"{'(ar)' if atom.aromatic else ''}, {atom.num_h} H"
                )
        return "\n".join(lines)


def annotate(
    tree: FragmentTree, *, atoms: bool = False, namer: Namer = name_fragment
) -> TreeAnnotation:
    """Describe every node in a fragment tree.

    Args:
        tree: The tree to describe.
        atoms: Include per-atom detail on each node (for modify/grow).
        namer: Resolves a fragment's common name; returns None when unknown.

    Returns:
        A serializable description of the tree.
    """
    return TreeAnnotation(
        nodes=[_annotate_node(tree, node, atoms, namer) for node in tree.nodes]
    )


def _annotate_node(
    tree: FragmentTree, node: FragmentNode, atoms: bool, namer: Namer
) -> NodeAnnotation:
    fragment = node.current
    mol = fragment.mol
    ports = [port.label for port in fragment.ports]
    neighbors = [
        NeighborRef(node_id=_id(other), port=edge.label, name=namer(other.current))
        for edge, other in tree.neighbors(node)
    ]
    return NodeAnnotation(
        id=_id(node),
        smiles=fragment.smiles,
        formula=_formula(mol),
        classification=classify_fragment(fragment),
        role="leaf" if len(ports) <= 1 else "scaffold",
        is_ring=any(a.IsInRing() for a in mol.GetAtoms()),
        name=namer(fragment),
        ports=ports,
        neighbors=neighbors,
        atoms=_atom_annotations(mol) if atoms else None,
    )


def _id(node: FragmentNode) -> int:
    """A tree-registered node's id; the annotation invariant is it has one."""
    assert node.id is not None, "node is not part of a tree"
    return node.id


def _atom_annotations(mol: Chem.Mol) -> list[AtomAnnotation]:
    return [
        AtomAnnotation(
            index=a.GetIdx(),
            element=a.GetSymbol(),
            aromatic=a.GetIsAromatic(),
            in_ring=a.IsInRing(),
            num_h=a.GetTotalNumHs(includeNeighbors=True),
        )
        for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1
    ]


def _formula(mol: Chem.Mol) -> str:
    """Hill-notation formula of the fragment, excluding port dummies."""
    counts: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        counts[atom.GetSymbol()] += 1
        counts["H"] += atom.GetNumImplicitHs()
    return _hill(counts)


def _hill(counts: Counter[str]) -> str:
    parts = []
    for symbol in ("C", "H"):
        n = counts.pop(symbol, 0)
        if n:
            parts.append(symbol + (str(n) if n > 1 else ""))
    for symbol in sorted(counts):
        n = counts[symbol]
        parts.append(symbol + (str(n) if n > 1 else ""))
    return "".join(parts)
