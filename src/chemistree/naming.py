"""Fragment naming and classification for annotations.

``name_fragment`` gives a specific common name when the fragment is in a small
curated table; ``classify_fragment`` always returns a coarse chemical class, so a
fragment never lacks descriptive context even when it has no curated name.
"""

from __future__ import annotations

from rdkit import Chem

from chemistree.fragment import Fragment

# Curated common fragments: SMILES (port as bare ``*``) -> name. Keys are
# canonicalized at import so lookups are label- and layout-independent. Extend
# freely; this is a reliability aid, not an exhaustive nomenclature.
_RAW_NAMES = {
    "*C": "methyl",
    "*CC": "ethyl",
    "*CCC": "propyl",
    "*C(C)C": "isopropyl",
    "*C(C)(C)C": "tert-butyl",
    "*C(F)(F)F": "trifluoromethyl",
    "*F": "fluoro",
    "*Cl": "chloro",
    "*Br": "bromo",
    "*I": "iodo",
    "*O": "hydroxyl",
    "*N": "amino",
    "*C#N": "cyano",
    "*[N+](=O)[O-]": "nitro",
    "*C=O": "formyl",
    "*C(=O)O": "carboxyl",
    "*c1ccccc1": "phenyl",
}
_NAMES = {Chem.CanonSmiles(smi): name for smi, name in _RAW_NAMES.items()}


def name_fragment(fragment: Fragment) -> str | None:
    """Curated common name for a fragment, or None if unrecognized."""
    return _NAMES.get(_canonical_key(fragment.mol))


def classify_fragment(fragment: Fragment) -> str:
    """Coarse chemical class from atom composition.

    Returns one of ``aromatic``, ``heteroaromatic``, ``alkyl``, or ``other``.
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
