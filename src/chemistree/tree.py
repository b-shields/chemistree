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
        """Record a new snapshot as the current fragment.

        Args:
            fragment: The snapshot to make current.
        """
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
        """Look up a node by its stable id.

        Args:
            node_id: The id assigned when the tree was built.

        Returns:
            The node with that id.
        """
        return self._by_id[node_id]

    def neighbors(self, node: FragmentNode) -> list[tuple[Edge, FragmentNode]]:
        """Edges incident to a node, paired with the node on the other side.

        Args:
            node: The node whose incident edges to return.

        Returns:
            An ``(edge, other node)`` pair for each edge touching ``node``.
        """
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

    def remove_subtree(self, node: FragmentNode) -> None:
        """Prune a node and its dependent groups, keeping the largest remainder.

        Cutting a node splits the tree into one component per edge. The largest
        component is kept; the node and every smaller component are removed. So
        removing a leaf drops just that leaf, and removing a ring drops the ring
        with its own substituents, while the main scaffold stays. The kept side's
        port to ``node`` is capped: its dummy becomes an explicit H, preserving
        the anchor's valence and 3D position.

        Args:
            node: The node to remove, with the smaller side(s) that depend on it.

        Raises:
            ValueError: If it is the only node, so nothing would remain.
        """
        components = self._components(without=node)
        if not components:
            raise ValueError("cannot remove the only fragment")
        largest = components[0]
        for component in components:
            if len(component) > len(largest):
                largest = component
        keep = set(largest)
        for edge, other in self.neighbors(node):
            if other in keep:
                _cap_port(other, edge.label)
                break
        removed = {node}
        for component in components:
            if not keep.issuperset(component):
                removed.update(component)
        self.nodes = [n for n in self.nodes if n not in removed]
        self.edges = [
            e for e in self.edges if e.node_a not in removed and e.node_b not in removed
        ]
        for gone in removed:
            if gone.id is not None:
                self._by_id.pop(gone.id, None)

    def _components(self, *, without: FragmentNode) -> list[list[FragmentNode]]:
        """Connected components of the tree with one node excluded."""
        seen = {without}
        components = []
        for start in self.nodes:
            if start in seen:
                continue
            component = []
            queue = [start]
            seen.add(start)
            while queue:
                current = queue.pop(0)
                component.append(current)
                for _, other in self.neighbors(current):
                    if other not in seen:
                        seen.add(other)
                        queue.append(other)
            components.append(component)
        return components

    def annotations(self, *, atoms: bool = False) -> TreeAnnotation:
        """A serializable, agent-facing description of the tree.

        Args:
            atoms: Include per-atom detail on each node.

        Returns:
            The tree annotation.
        """
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


def _cap_port(node: FragmentNode, label: int) -> None:
    """Turn a node's dummy port into hydrogen, pushing the capped fragment.

    Converting the dummy to H (rather than deleting it) keeps the anchor's
    valence and the dummy's 3D position. The change is pushed as a new snapshot
    so it can be reverted like any other edit.

    Args:
        node: The node whose port is capped.
        label: Isotope label of the dummy port to cap.
    """
    mol = Chem.RWMol(node.current.mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == label:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            break
    capped = mol.GetMol()
    Chem.SanitizeMol(capped)
    node.push(Fragment(capped))
