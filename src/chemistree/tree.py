"""Fragment tree: nodes, port-paired edges, and molecule reconstruction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rdkit import Chem

from chemistree.fragment import Fragment


class FragmentNode:
    """A tree node holding a stack of fragment snapshots (top = current)."""

    def __init__(self, fragment: Fragment):
        self.history: list[Fragment] = [fragment]

    @property
    def current(self) -> Fragment:
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

    def leaves(self) -> list[FragmentNode]:
        """Nodes with at most one edge."""
        degree: dict[int, int] = defaultdict(int)
        for edge in self.edges:
            degree[id(edge.node_a)] += 1
            degree[id(edge.node_b)] += 1
        return [n for n in self.nodes if degree[id(n)] <= 1]

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
