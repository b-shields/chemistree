"""Deterministic positional selection over a fragment.

Topological distance is the universal primitive: it addresses positions on rings
and chains alike. ortho/meta/para are a ring-only synonym shim over distances
1/2/3; ``para`` is defined only on 6-membered rings.
"""

from __future__ import annotations

from typing import Callable

from rdkit import Chem

from chemistree.naming import classify_fragment, name_fragment
from chemistree.tree import FragmentNode, FragmentTree

POSITION_SYNONYMS = {"ortho": 1, "meta": 2, "para": 3}

SitePredicate = Callable[[Chem.Mol, int], bool]


class ResolutionError(Exception):
    """A reference could not be resolved to a single node or atom."""


class NotFound(ResolutionError):
    """No candidate matched the reference."""


class Ambiguous(ResolutionError):
    """More than one candidate matched the reference.

    Attributes:
        candidates: Ids of the nodes that matched.
    """

    def __init__(self, message: str, candidates: list[int]):
        super().__init__(message)
        self.candidates = candidates


def select(
    tree: FragmentTree,
    *,
    name: str | None = None,
    classification: str | None = None,
    neighbor_of: FragmentNode | None = None,
) -> list[FragmentNode]:
    """Nodes matching every given constraint.

    Args:
        tree: The tree to search.
        name: Required common name (see ``name_fragment``).
        classification: Required coarse class (see ``classify_fragment``).
        neighbor_of: Keep only nodes adjacent to this node.

    Returns:
        The matching nodes, in tree order.
    """
    adjacent = (
        {other for _, other in tree.neighbors(neighbor_of)}
        if neighbor_of is not None
        else None
    )
    matches = []
    for node in tree.nodes:
        fragment = node.current
        if name is not None and name_fragment(fragment) != name:
            continue
        if classification is not None and classify_fragment(fragment) != classification:
            continue
        if adjacent is not None and node not in adjacent:
            continue
        matches.append(node)
    return matches


def select_one(
    tree: FragmentTree,
    *,
    description: str = "node",
    name: str | None = None,
    classification: str | None = None,
    neighbor_of: FragmentNode | None = None,
) -> FragmentNode:
    """The single node matching the constraints, or a structured error.

    Args:
        tree: The tree to search.
        description: Noun used in error messages (e.g. "pyridine").
        name: Required common name.
        classification: Required coarse class.
        neighbor_of: Keep only nodes adjacent to this node.

    Returns:
        The unique matching node.

    Raises:
        NotFound: If no node matches.
        Ambiguous: If more than one node matches.
    """
    matches = select(
        tree, name=name, classification=classification, neighbor_of=neighbor_of
    )
    if not matches:
        raise NotFound(f"no {description} found")
    if len(matches) > 1:
        raise Ambiguous(
            f"{len(matches)} candidates match {description}",
            [n.id for n in matches if n.id is not None],
        )
    return matches[0]


def attachment_atom(
    tree: FragmentTree, scaffold: FragmentNode, substituent: FragmentNode
) -> int:
    """The scaffold atom that bears a given substituent.

    Args:
        tree: The tree the nodes belong to.
        scaffold: The node carrying the substituent.
        substituent: The adjacent node.

    Returns:
        Index (in the scaffold's current fragment) of the atom the substituent
        attaches to.

    Raises:
        NotFound: If the substituent is not attached to the scaffold.
    """
    for edge, other in tree.neighbors(scaffold):
        if other is substituent:
            for port in scaffold.current.ports:
                if port.label == edge.label:
                    return port.anchor_idx
    raise NotFound("substituent is not attached to scaffold")


def resolve_offset(spec: int | str, ring_size: int | None = None) -> int:
    """Translate a position spec to a topological distance.

    An integer passes through unchanged. A synonym (ortho/meta/para) maps to a
    distance; ``para`` requires a 6-membered ring, since distance 3 is the unique
    opposite position only there.

    Args:
        spec: A bond count, or one of ``ortho`` / ``meta`` / ``para``.
        ring_size: Size of the ring the reference sits in, needed to validate
            ``para``.

    Returns:
        The topological distance in bonds.

    Raises:
        ValueError: If the term is unknown, or ``para`` is used off a 6-ring.
    """
    if isinstance(spec, int):
        return spec
    if spec not in POSITION_SYNONYMS:
        raise ValueError(f"unknown position term: {spec!r}")
    if spec == "para" and ring_size != 6:
        raise ValueError("'para' is only defined on 6-membered rings")
    return POSITION_SYNONYMS[spec]


def atoms_at_distance(mol: Chem.Mol, source: int, distance: int) -> list[int]:
    """Heavy atoms exactly ``distance`` bonds from a source atom.

    Distance is the shortest path in the molecular graph, so it counts the same on
    rings and chains. Hydrogens and port dummies are excluded from the result.

    Args:
        mol: The fragment molecule.
        source: Index of the reference atom.
        distance: Number of bonds away.

    Returns:
        Indices of the matching heavy atoms.
    """
    dmat = Chem.GetDistanceMatrix(mol)
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and int(dmat[source][atom.GetIdx()]) == distance
    ]


def is_open_position(mol: Chem.Mol, atom: int) -> bool:
    """Whether an atom can host a new substituent.

    An open position has a hydrogen to replace and no port (dummy) already attached.

    Args:
        mol: The fragment molecule.
        atom: Index of the candidate atom.

    Returns:
        True if the atom is a free growth site.
    """
    target = mol.GetAtomWithIdx(atom)
    if target.GetTotalNumHs(includeNeighbors=True) < 1:
        return False
    return not any(neighbor.GetAtomicNum() == 0 for neighbor in target.GetNeighbors())


def is_free_aromatic_carbon(mol: Chem.Mol, atom: int) -> bool:
    """Whether an atom is an open aromatic-carbon position (excludes ring N, O, S).

    Args:
        mol: The fragment molecule.
        atom: Index of the candidate atom.

    Returns:
        True if the atom is aromatic carbon and a free growth site.
    """
    target = mol.GetAtomWithIdx(atom)
    return (
        target.GetIsAromatic()
        and target.GetAtomicNum() == 6
        and is_open_position(mol, atom)
    )


def resolve_position(
    mol: Chem.Mol,
    reference: int,
    offset: int | str,
    *,
    site: SitePredicate = is_open_position,
) -> int:
    """Resolve a position ``offset`` bonds from a reference atom to one atom.

    This is the universal step: it counts topological distance, so it addresses
    positions on rings (with ortho/meta/para) and chains ("two carbons in") alike.

    Args:
        mol: The fragment molecule.
        reference: Index of the atom to count from.
        offset: A bond count, or a ring synonym (ortho/meta/para).
        site: Constraint the target atom must satisfy.

    Returns:
        Index of the single matching atom.

    Raises:
        NotFound: If no atom at that offset satisfies the constraint.
        Ambiguous: If more than one does.
        ValueError: If a ring synonym is invalid for the reference's ring.
    """
    distance = resolve_offset(offset, ring_size=_ring_size_at(mol, reference))
    candidates = [
        a for a in atoms_at_distance(mol, reference, distance) if site(mol, a)
    ]
    if not candidates:
        raise NotFound(f"no open position {offset} from the reference")
    if len(candidates) > 1:
        raise Ambiguous(
            f"{len(candidates)} positions {offset} from the reference", candidates
        )
    return candidates[0]


def resolve_site(
    tree: FragmentTree,
    scaffold: FragmentNode,
    reference: FragmentNode,
    position: int | str,
    *,
    site: SitePredicate = is_open_position,
) -> int:
    """Resolve a position on a scaffold relative to one of its substituents.

    Args:
        tree: The tree the nodes belong to.
        scaffold: The node whose atom to resolve.
        reference: A substituent of the scaffold to count from.
        position: A bond count, or a ring synonym (ortho/meta/para).
        site: Constraint the target atom must satisfy.

    Returns:
        Index (in the scaffold's current fragment) of the single matching atom.

    Raises:
        NotFound: If nothing matches. Ambiguous: If several do.
    """
    anchor = attachment_atom(tree, scaffold, reference)
    return resolve_position(scaffold.current.mol, anchor, position, site=site)


def _ring_size_at(mol: Chem.Mol, atom: int) -> int | None:
    """Size of the smallest ring containing an atom, or None if it is acyclic."""
    sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings() if atom in ring]
    return min(sizes) if sizes else None
