"""Fragment tree: nodes, port-paired edges, and molecule reconstruction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rdkit import Chem

from chemistree.fragment import Fragment

if TYPE_CHECKING:
    from chemistree.annotations import TreeAnnotation


class FragmentNode:
    """A tree node holding a stack of fragment snapshots (top = current)."""

    def __init__(self, fragment: Fragment):
        self.history: list[Fragment] = [fragment]
        self.id: int | None = None  # assigned by the owning FragmentTree

    @property
    def current(self) -> Fragment:
        """The current (top-of-stack) fragment snapshot."""
        return self.history[-1]

    def push(self, fragment: Fragment) -> None:
        """Record a new snapshot as the current fragment."""
        self.history.append(fragment)

    def undo(self) -> None:
        """Revert to the previous snapshot, if any."""
        if len(self.history) > 1:
            self.history.pop()

    def __repr__(self) -> str:
        return f"FragmentNode({self.current!r})"


@dataclass
class Edge:
    """A broken bond linking two nodes at their matching ports.

    Attributes:
        label: The port pairing label shared by both sides.
        node_a: One endpoint node.
        node_b: The other endpoint node.
        bond_type: Bond order to restore on reconstruction.
    """

    label: int
    node_a: FragmentNode
    node_b: FragmentNode
    bond_type: Chem.BondType


@dataclass
class FragmentTree:
    """An unrooted free tree of fragments connected by port-paired edges."""

    nodes: list[FragmentNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._next_id = 0
        self._by_id: dict[int, FragmentNode] = {}
        for node in self.nodes:
            self._register(node)

    def _register(self, node: FragmentNode) -> None:
        """Assign a stable id so nodes can be referenced across edits."""
        node.id = self._next_id
        self._by_id[node.id] = node
        self._next_id += 1

    def node(self, node_id: int) -> FragmentNode:
        """Look up a node by its stable id."""
        return self._by_id[node_id]

    def neighbors(self, node: FragmentNode) -> list[tuple[Edge, FragmentNode]]:
        """Edges incident to ``node`` paired with the node on the other side."""
        out = []
        for edge in self.edges:
            if edge.node_a is node:
                out.append((edge, edge.node_b))
            elif edge.node_b is node:
                out.append((edge, edge.node_a))
        return out

    def leaves(self) -> list[FragmentNode]:
        """Nodes with at most one edge."""
        return [n for n in self.nodes if len(self.neighbors(n)) <= 1]

    def annotations(self, *, atoms: bool = False) -> TreeAnnotation:
        """A serializable, agent-facing description of the tree."""
        from chemistree.annotations import annotate

        return annotate(self, atoms=atoms)

    def reconstruct(self) -> Chem.Mol:
        """Fuse all current fragments back into a single molecule.

        Combines every node's current fragment, then for each edge bonds the two
        anchor atoms and deletes the dummy atoms that marked the port.
        """
        combined = None
        for node in self.nodes:
            mol = node.current.mol
            combined = mol if combined is None else Chem.CombineMols(combined, mol)

        rw = Chem.RWMol(combined)
        bond_types = {edge.label: edge.bond_type for edge in self.edges}

        dummies: dict[int, list[int]] = defaultdict(list)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummies[atom.GetIsotope()].append(atom.GetIdx())

        to_remove = []
        for label, (d1, d2) in dummies.items():
            a1 = rw.GetAtomWithIdx(d1).GetNeighbors()[0].GetIdx()
            a2 = rw.GetAtomWithIdx(d2).GetNeighbors()[0].GetIdx()
            rw.AddBond(a1, a2, bond_types[label])
            to_remove += [d1, d2]

        for idx in sorted(to_remove, reverse=True):
            rw.RemoveAtom(idx)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol
