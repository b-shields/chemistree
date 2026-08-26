"""Fragment tree: nodes, port-paired edges, and molecule reconstruction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
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
class RemovedSubtree:
    """What a ``remove_subtree`` took out, enough to put it back.

    Attributes:
        nodes: The removed nodes (the target and its dependent groups).
        edges: The edges that were incident to any removed node.
        parent: The kept node whose port was capped, to be uncapped on restore.
        parent_site: Index (in ``parent``'s fragment) of the atom whose port was
            capped, so a later edit can grow a group back at the freed site.
    """

    nodes: list[FragmentNode]
    edges: list[Edge]
    parent: FragmentNode
    parent_site: int


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

    def resplice(
        self, node: FragmentNode, region_mol: Chem.Mol
    ) -> tuple[list[FragmentNode], Callable[[], None]]:
        """Replace a node with the fragmentation of its edited region.

        ``region_mol`` is the node's atoms after an edit, still carrying the node's
        existing external-port dummies. It is re-fragmented with the same
        ``should_break`` policy that construction uses, so the tree stays identical
        (up to node numbering) to fragmenting the whole edited molecule: a new node
        appears only where the policy breaks a bond, and the group merges otherwise.
        External ports reconnect to the neighbors they had; new internal ports pair
        the resulting sub-fragments. The largest sub-fragment keeps ``node``'s id.

        Args:
            node: The node whose region was edited.
            region_mol: The edited region, with the node's external-port dummies.

        Returns:
            The new sub-nodes and an undo closure that restores the prior tree.
        """
        from chemistree.fragmenter import should_break

        before = (list(self.nodes), list(self.edges), dict(self._by_id), self._next_id)
        external = {edge.label: other for edge, other in self.neighbors(node)}

        breakable = [
            b.GetIdx() for b in region_mol.GetBonds() if should_break(region_mol, b)
        ]
        if breakable:
            labels = list(
                range(self._max_label() + 1, self._max_label() + 1 + len(breakable))
            )
            broken = Chem.FragmentOnBonds(
                region_mol,
                breakable,
                addDummies=True,
                dummyLabels=[(la, la) for la in labels],
            )
            pieces = list(Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True))
        else:
            pieces = [region_mol]

        sub_nodes = [FragmentNode(Fragment(p)) for p in pieces]
        self._replace_node(node, sub_nodes)
        self._rewire(sub_nodes, external)

        def undo() -> None:
            """Restore the tree to its state before this resplice."""
            self.nodes, self.edges, self._by_id, self._next_id = (
                before[0],
                before[1],
                before[2],
                before[3],
            )

        return sub_nodes, undo

    def _replace_node(self, node: FragmentNode, sub_nodes: list[FragmentNode]) -> None:
        """Swap ``node`` out for its sub-nodes; the largest inherits its id."""
        self.nodes = [n for n in self.nodes if n is not node]
        self.edges = [
            e for e in self.edges if e.node_a is not node and e.node_b is not node
        ]
        if node.id is not None:
            self._by_id.pop(node.id, None)
        core = max(sub_nodes, key=lambda n: heavy_count(n.current.mol))
        for sub in sub_nodes:
            if sub is core and node.id is not None:
                sub.id = node.id
                self._by_id[node.id] = sub
            else:
                self._register(sub)
            self.nodes.append(sub)

    def _rewire(
        self, sub_nodes: list[FragmentNode], external: dict[int, FragmentNode]
    ) -> None:
        """Add edges: external labels to old neighbors, new labels between sub-nodes."""
        carriers: dict[int, list[FragmentNode]] = defaultdict(list)
        for sub in sub_nodes:
            for port in sub.current.ports:
                carriers[port.label].append(sub)
        for label, subs in carriers.items():
            if label in external:
                self.edges.append(
                    Edge(label, subs[0], external[label], Chem.BondType.SINGLE)
                )
            else:
                self.edges.append(Edge(label, subs[0], subs[1], Chem.BondType.SINGLE))

    def _max_label(self) -> int:
        """The largest port label anywhere in the tree, or 0 if there are none."""
        labels = [port.label for n in self.nodes for port in n.current.ports]
        return max(labels, default=0)

    def leaves(self) -> list[FragmentNode]:
        """Nodes with at most one edge."""
        return [n for n in self.nodes if len(self.neighbors(n)) <= 1]

    def remove_subtree(self, node: FragmentNode) -> RemovedSubtree:
        """Prune a node and its dependent groups, keeping the largest remainder.

        Cutting a node splits the tree into one component per edge. The largest
        component is kept; the node and every smaller component are removed. So
        removing a leaf drops just that leaf, and removing a ring drops the ring
        with its own substituents, while the main scaffold stays. The kept side's
        port to ``node`` is capped: its dummy becomes an explicit H, preserving
        the anchor's valence and 3D position.

        Args:
            node: The node to remove, with the smaller side(s) that depend on it.

        Returns:
            A record of what was removed, so ``restore_subtree`` can undo it.

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
        parent = None
        parent_site = -1
        for edge, other in self.neighbors(node):
            if other in keep:
                parent = other
                # Record the anchor before capping, so a later ``fill`` knows
                # which atom the freed port sat on.
                parent_site = next(
                    p.anchor_idx for p in other.current.ports if p.label == edge.label
                )
                _cap_port(other, edge.label)
                break
        assert parent is not None  # a non-only node always touches the kept side
        removed = {node}
        for component in components:
            if not keep.issuperset(component):
                removed.update(component)
        removed_edges = [
            e for e in self.edges if e.node_a in removed or e.node_b in removed
        ]
        self.nodes = [n for n in self.nodes if n not in removed]
        self.edges = [e for e in self.edges if e not in removed_edges]
        for gone in removed:
            if gone.id is not None:
                self._by_id.pop(gone.id, None)
        return RemovedSubtree(
            nodes=list(removed),
            edges=removed_edges,
            parent=parent,
            parent_site=parent_site,
        )

    def restore_subtree(self, removed: RemovedSubtree) -> None:
        """Undo a ``remove_subtree``: re-add its nodes and edges, uncap the parent.

        Args:
            removed: The record returned by ``remove_subtree``.
        """
        removed.parent.undo()  # revert the cap, restoring the dummy port
        for node in removed.nodes:
            self.nodes.append(node)
            if node.id is not None:
                self._by_id[node.id] = node
        self.edges.extend(removed.edges)

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
        anchor atoms and deletes the dummy atoms that marked the port. Every atom
        carries a ``node_id`` int property naming the group it came from, so a
        caller can attribute an atom in the whole molecule back to its group.
        """
        if not self.nodes:
            raise ValueError("cannot reconstruct an empty tree")
        tagged = []
        for node in self.nodes:
            piece = Chem.Mol(node.current.mol)
            for atom in piece.GetAtoms():
                atom.SetIntProp("node_id", node.id if node.id is not None else -1)
            tagged.append(piece)
        combined = tagged[0]
        for piece in tagged[1:]:
            combined = Chem.CombineMols(combined, piece)

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


def heavy_count(mol: Chem.Mol) -> int:
    """Number of heavy (non-hydrogen, non-dummy) atoms in a molecule."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


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
