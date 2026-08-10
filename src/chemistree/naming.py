"""Fragment naming and classification for annotations.

``name_fragment`` gives a specific common name when the fragment is in a small
curated table; ``classify_fragment`` always returns a coarse chemical class, so a
fragment never lacks descriptive context even when it has no curated name.
"""

from __future__ import annotations

from rdkit import Chem

from chemistree.fragment import Fragment
from chemistree.functional_groups import functional_group_name

# Curated common fragments: SMILES (port as bare ``*``) -> name. Keys are
# canonicalized at import so lookups are label- and layout-independent. Extend
# freely; this is a reliability aid, not an exhaustive nomenclature.
_RAW_NAMES = {
    # Alkyl / cycloalkyl (terminal)
    "*C": "methyl",
    "*CC": "ethyl",
    "*CCC": "propyl",
    "*C(C)C": "isopropyl",
    "*CCCC": "butyl",
    "*C(C)(C)C": "tert-butyl",
    "*C1CC1": "cyclopropyl",
    "*C1CCC1": "cyclobutyl",
    "*C1CCCC1": "cyclopentyl",
    "*C1CCCCC1": "cyclohexyl",
    # Halogens
    "*F": "fluoro",
    "*Cl": "chloro",
    "*Br": "bromo",
    "*I": "iodo",
    # Fluorinated
    "*C(F)F": "difluoromethyl",
    "*C(F)(F)F": "trifluoromethyl",
    "*OC(F)(F)F": "trifluoromethoxy",
    # Oxygen (terminal)
    "*O": "hydroxyl",
    "*OC": "methoxy",
    "*OCC": "ethoxy",
    # Sulfur (terminal)
    "*S": "thiol",
    "*SC": "methylthio",
    # Nitrogen (terminal)
    "*N": "amino",
    "*NC": "methylamino",
    "*N(C)C": "dimethylamino",
    # Unsaturated (terminal)
    "*C=C": "vinyl",
    "*C#C": "ethynyl",
    # Nitrile / nitro
    "*C#N": "cyano",
    "*[N+](=O)[O-]": "nitro",
    # Carbonyl-derived (terminal)
    "*C=O": "formyl",
    "*C(C)=O": "acetyl",
    "*C(=O)O": "carboxyl",
    "*C(=O)OC": "methoxycarbonyl",
    "*C(N)=O": "carboxamide",
    # Oxidized sulfur (terminal)
    "*S(=O)(=O)O": "sulfo",
    "*S(N)(=O)=O": "sulfamoyl",
    "*S(C)(=O)=O": "methylsulfonyl",
    # Aryl (terminal). Heteroaryls and multi-substituted rings are position-
    # dependent; those fall back to `classify_fragment` until ring-system naming.
    "*c1ccccc1": "phenyl",
    # Two-port linkers (what fragmentation actually yields for internal groups)
    "*C*": "methylene",
    "*CC*": "ethylene",
    "*O*": "ether",
    "*S*": "thioether",
    "*N*": "amine",
    "*C(*)=O": "carbonyl",
    "*C(=O)O*": "ester",
    "*C(=O)N*": "amide",
    "*S(*)(=O)=O": "sulfonyl",
    "*S(=O)(=O)N*": "sulfonamide",
    "*N=N*": "azo",
}
_NAMES = {Chem.CanonSmiles(smi): name for smi, name in _RAW_NAMES.items()}
_GROUP_SMILES = {name: smi for smi, name in _RAW_NAMES.items()}


def group_smiles(name: str) -> str | None:
    """SMILES for a curated group name (the reverse of the naming table).

    Args:
        name: A common group name, e.g. "isopropyl".

    Returns:
        The group's SMILES with a bare ``*`` port, or None if the name is unknown.
    """
    return _GROUP_SMILES.get(name)


# Ring systems named by substructure, independent of substitution or port count.
# Ordered fused-first; a candidate names a fragment only when it covers exactly the
# fragment's ring atoms, so benzene never mis-names an indole.
_RING_SYSTEM_SMILES = [
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccc2ncccc2c1", "quinoline"),
    ("c1ccc2[nH]ccc2c1", "indole"),
    ("c1ccc2[nH]cnc2c1", "benzimidazole"),
    ("c1ccc2occc2c1", "benzofuran"),
    ("c1ccc2sccc2c1", "benzothiophene"),
    ("c1ccncc1", "pyridine"),
    ("c1cncnc1", "pyrimidine"),
    ("c1cnccn1", "pyrazine"),
    ("c1ccnnc1", "pyridazine"),
    ("c1ccccc1", "benzene"),
    ("c1cc[nH]c1", "pyrrole"),
    ("c1ccoc1", "furan"),
    ("c1ccsc1", "thiophene"),
    ("c1c[nH]cn1", "imidazole"),
    ("c1cc[nH]n1", "pyrazole"),
    ("c1ocnc1", "oxazole"),
    ("c1ccon1", "isoxazole"),
    ("c1cscn1", "thiazole"),
    ("c1nc[nH]n1", "triazole"),
    ("C1CCNCC1", "piperidine"),
    ("C1CNCCN1", "piperazine"),
    ("C1COCCN1", "morpholine"),
    ("C1CCNC1", "pyrrolidine"),
    ("C1CCOC1", "tetrahydrofuran"),
    ("C1CCOCC1", "tetrahydropyran"),
    ("C1CC1", "cyclopropane"),
    ("C1CCC1", "cyclobutane"),
    ("C1CCCC1", "cyclopentane"),
    ("C1CCCCC1", "cyclohexane"),
]
_RING_SYSTEMS = [(Chem.MolFromSmiles(smi), name) for smi, name in _RING_SYSTEM_SMILES]


def name_fragment(fragment: Fragment) -> str | None:
    """Common name for a fragment.

    Tries, in order: an exact curated substituent match, the fragment's ring
    system, then the functional group it contains.

    Args:
        fragment: The fragment to name.

    Returns:
        The common name, or None if no strategy recognizes it.
    """
    exact = _NAMES.get(_canonical_key(fragment.mol))
    if exact is not None:
        return exact
    ring = _ring_system_name(fragment.mol)
    if ring is not None:
        return ring
    return functional_group_name(fragment.mol)


def _ring_system_name(mol: Chem.Mol) -> str | None:
    """Name a fragment's ring system when every heavy atom is a ring atom.

    Only bare ring systems are named (all non-port, non-hydrogen atoms in rings).
    The matching candidate must cover them all, so ports and substitution don't
    affect the result and a sub-ring never names a larger fused system.
    """
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if not heavy or not all(mol.GetAtomWithIdx(i).IsInRing() for i in heavy):
        return None
    for pattern, name in _RING_SYSTEMS:
        if any(frozenset(m) == heavy for m in mol.GetSubstructMatches(pattern)):
            return name
    return None


def classify_fragment(fragment: Fragment) -> str:
    """Coarse chemical class from atom composition.

    Args:
        fragment: The fragment to classify.

    Returns:
        One of ``aromatic``, ``heteroaromatic``, ``alkyl``, or ``other``.
    """
    heavy = [a for a in fragment.mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return "other"
    aromatic = [a for a in heavy if a.GetIsAromatic()]
    if aromatic:
        if any(a.GetAtomicNum() != 6 for a in aromatic):
            return "heteroaromatic"
        return "aromatic"
    if all(a.GetAtomicNum() == 6 for a in heavy):
        return "alkyl"
    return "other"


def _canonical_key(mol: Chem.Mol) -> str:
    """Canonical SMILES with ports normalized to bare ``*`` and Hs implicit."""
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return str(Chem.MolToSmiles(Chem.RemoveHs(mol)))
