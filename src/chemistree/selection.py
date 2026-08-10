"""Deterministic positional selection over a fragment.

Topological distance is the universal primitive: it addresses positions on rings
and chains alike. ortho/meta/para are a ring-only synonym shim over distances
1/2/3; ``para`` is defined only on 6-membered rings.
"""

from __future__ import annotations

from rdkit import Chem

POSITION_SYNONYMS = {"ortho": 1, "meta": 2, "para": 3}


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
