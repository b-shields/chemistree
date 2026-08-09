"""Bond-breaking policy and tree construction.

The only refinable surface is :func:`should_break`. Everything above it is a
fixed driver: because we only ever break bridge (acyclic single) bonds, the
resulting fragment graph is a tree independent of break order.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from rdkit import Chem

from chemistree.fragment import Fragment
from chemistree.functional_groups import protects_bond
from chemistree.tree import Edge, FragmentNode, FragmentTree

BondSelector = Callable[[Chem.Mol, Chem.Bond], bool]

# Functional groups whose bonds to the rest of the molecule are breakable. This
# is the primary place to refine fragmentation granularity.
_FG_SMARTS = (
    "[CX3]=[OX1]",  # carbonyl (acids, esters, amides, ketones, aldehydes)
    "[NX3](=[OX1])=[OX1]",  # nitro
    "[SX4](=[OX1])(=[OX1])",  # sulfonyl
    "[SX3](=[OX1])",  # sulfinyl
    "[PX4](=[OX1])",  # phosphoryl
)
_FG_PATTERNS = [Chem.MolFromSmarts(s) for s in _FG_SMARTS]


def _fg_atoms(mol: Chem.Mol) -> set[int]:
    """Indices of atoms belonging to any matched functional group."""
    atoms: set[int] = set()
    for pattern in _FG_PATTERNS:
        for match in mol.GetSubstructMatches(pattern):
            atoms.update(match)
    return atoms


def should_break(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """Whether a bond should be broken during fragmentation.

    Breaks a single, acyclic bond when either endpoint is a ring atom or the bond
    is a functional-group boundary (exactly one endpoint inside a matched group).
    Bonds internal to a recognized functional group are kept, so groups like
    esters and amides stay intact as single, named units.

    Args:
        mol: The molecule the bond belongs to.
        bond: The candidate bond.

    Returns:
        True if the bond should be broken.
    """
    if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
        return False

    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    if a.GetAtomicNum() <= 1 or b.GetAtomicNum() <= 1:
        return False  # never break bonds to ports (dummies) or hydrogens

    if protects_bond(mol, bond):
        return False  # keep functional groups intact

    if a.IsInRing() or b.IsInRing():
        return True

    fg = _fg_atoms(mol)
    return (a.GetIdx() in fg) != (b.GetIdx() in fg)


def fragment(mol: Chem.Mol, selector: BondSelector = should_break) -> FragmentTree:
    """Fragment a molecule into a tree by breaking all selected bonds.

    Args:
        mol: The molecule to fragment.
        selector: Predicate deciding which bonds to break.

    Returns:
        The resulting fragment tree.
    """
    mol = Chem.Mol(mol)

    # Identify breakable bonds
    bonds = [b.GetIdx() for b in mol.GetBonds() if selector(mol, b)]
    if not bonds:
        return FragmentTree(nodes=[FragmentNode(Fragment(mol))], edges=[])

    # Fragment on breakable bonds (coordinate preserving)
    labels = list(range(1, len(bonds) + 1))
    broken = Chem.FragmentOnBonds(
        mol, bonds, addDummies=True, dummyLabels=[(la, la) for la in labels]
    )
    pieces = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    nodes = [FragmentNode(Fragment(p)) for p in pieces]

    # Build nodes
    label_to_nodes: dict[int, list[FragmentNode]] = defaultdict(list)
    for node in nodes:
        for port in node.current.ports:
            label_to_nodes[port.label].append(node)

    # Build edges
    edges = [
        Edge(label, na, nb, Chem.BondType.SINGLE)
        for label, (na, nb) in label_to_nodes.items()
    ]

    # Order core-first so the largest fragment gets id 0
    nodes.sort(key=_core_first)
    return FragmentTree(nodes=nodes, edges=edges)


def _core_first(node: FragmentNode) -> tuple[int, int, str]:
    """Sort key placing the largest, most-connected fragment first."""
    fragment = node.current
    heavy = sum(1 for a in fragment.mol.GetAtoms() if a.GetAtomicNum() > 1)
    return (-heavy, -len(fragment.ports), fragment.smiles)
