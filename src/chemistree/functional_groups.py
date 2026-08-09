"""Functional-group registry: SMARTS + name, one source of truth.

Used both to (a) protect a group from being split during fragmentation and (b)
name the resulting node. Only groups with an internal *single* bond need
protecting — double/triple-bonded groups (nitro, cyano, sulfonyl-to-oxygen, ...)
already survive fragmentation intact.

Ordered most-specific first so a containing group wins over its sub-motifs
(carbamate before ester + amide). Protection is order-independent (a bond is kept
if it is internal to *any* group); ordering only affects name selection.
"""

from __future__ import annotations

from rdkit import Chem

_FUNCTIONAL_GROUP_SMARTS = [
    ("[NX3][CX3](=[OX1])[OX2]", "carbamate"),
    ("[NX3][CX3](=[OX1])[NX3]", "urea"),
    ("[OX2][CX3](=[OX1])[OX2]", "carbonate"),
    ("[NX3][CX3](=[NX2])[NX3]", "guanidine"),
    ("[SX4](=[OX1])(=[OX1])[NX3]", "sulfonamide"),
    ("[NX3][CX3]=[NX2]", "amidine"),
    ("[CX3](=[OX1])[OX2]", "ester"),  # also matches carboxylic acid
    ("[CX3](=[OX1])[NX3]", "amide"),
]

FUNCTIONAL_GROUPS = [
    (Chem.MolFromSmarts(smarts), name) for smarts, name in _FUNCTIONAL_GROUP_SMARTS
]


def protects_bond(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """Whether a bond lies inside a functional group and must be kept intact.

    Args:
        mol: The molecule the bond belongs to.
        bond: The candidate bond.

    Returns:
        True if both of the bond's atoms fall inside a single functional group.
    """
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    for pattern, _ in FUNCTIONAL_GROUPS:
        if any(i in match and j in match for match in mol.GetSubstructMatches(pattern)):
            return True
    return False


def functional_group_name(mol: Chem.Mol) -> str | None:
    """Name a fragment by the most specific functional group it contains.

    Args:
        mol: The fragment molecule.

    Returns:
        The functional-group name, or None if it contains no registered group.
    """
    for pattern, name in FUNCTIONAL_GROUPS:
        if mol.HasSubstructMatch(pattern):
            return name
    return None
