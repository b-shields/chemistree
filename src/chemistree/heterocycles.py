"""Vendored parent heteroaromatic rings: name and canonical SMILES.

The single source of truth for the named rings the agent may ask about. Backs the
``matches`` tool, so a ring can be checked by name and the ring's name is echoed
back. The same list is shown to the agent in the guidance prompts; a test keeps the
two in sync. Extend freely.
"""

from __future__ import annotations

from rdkit import Chem

# name -> parent-ring SMILES (rdkit canonical). Names are lower-case for lookup.
HETEROCYCLES: dict[str, str] = {
    # 5-membered, one heteroatom
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "pyrrole": "c1cc[nH]c1",
    # 5-membered, two heteroatoms
    "pyrazole": "c1cn[nH]c1",
    "imidazole": "c1c[nH]cn1",
    "isoxazole": "c1cnoc1",
    "oxazole": "c1cocn1",
    "isothiazole": "c1cnsc1",
    "thiazole": "c1cscn1",
    # 5-membered, three or more heteroatoms
    "1,2,3-triazole": "c1cn[nH]n1",
    "1,2,4-triazole": "c1nc[nH]n1",
    "tetrazole": "c1nn[nH]n1",
    "1,2,4-oxadiazole": "c1ncon1",
    "1,3,4-oxadiazole": "c1nnco1",
    "1,3,4-thiadiazole": "c1nncs1",
    # 6-membered
    "pyridine": "c1ccncc1",
    "pyridazine": "c1ccnnc1",
    "pyrimidine": "c1cncnc1",
    "pyrazine": "c1cnccn1",
    "1,3,5-triazine": "c1ncncn1",
    # fused 5-6
    "indole": "c1ccc2[nH]ccc2c1",
    "indazole": "c1ccc2[nH]ncc2c1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "benzofuran": "c1ccc2occc2c1",
    "benzothiophene": "c1ccc2sccc2c1",
    "benzoxazole": "c1ccc2ocnc2c1",
    "benzothiazole": "c1ccc2scnc2c1",
    "7-azaindole": "c1cnc2[nH]ccc2c1",
    "purine": "c1ncc2[nH]cnc2n1",
    "pyrrolotriazine": "c1cc2cncnn2c1",  # pyrrolo[2,1-f][1,2,4]triazine
    # fused 6-6
    "quinoline": "c1ccc2ncccc2c1",
    "isoquinoline": "c1ccc2cnccc2c1",
    "quinazoline": "c1ccc2ncncc2c1",
    "quinoxaline": "c1ccc2nccnc2c1",
    "1,8-naphthyridine": "c1cnc2ncccc2c1",
}

# canonical SMILES -> name, built once for layout-independent reverse lookup.
_BY_CANONICAL: dict[str, str] = {}
for _name, _smiles in HETEROCYCLES.items():
    _mol = Chem.MolFromSmiles(_smiles)
    if _mol is not None:
        _BY_CANONICAL[Chem.MolToSmiles(_mol)] = _name


def heterocycle_smiles(name: str) -> str | None:
    """The parent-ring SMILES for a heterocycle name, or None if unknown.

    Args:
        name: A heterocycle name (case-insensitive), e.g. ``"quinazoline"``.

    Returns:
        The ring SMILES, or None when the name is not in the table.
    """
    return HETEROCYCLES.get(name.strip().lower())


def heterocycle_name(smiles: str) -> str | None:
    """The heterocycle name whose parent ring equals a SMILES, or None.

    Args:
        smiles: A SMILES string; canonicalized before the lookup.

    Returns:
        The matching heterocycle name, or None if it is not a known parent ring.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _BY_CANONICAL.get(Chem.MolToSmiles(mol))


# IUPAC ring positions, aligned to each ring's canonical SMILES atom order (0 = ring
# fusion or unnumbered). rdkit indexes atoms in SMILES-string order, so a plain tuple
# aligns without atom-map annotation. Every entry is verified against the depiction (a
# test locks it in). purine and pyrrolotriazine are deferred (no numbering yet), so
# number_ring_system returns None for them.
RING_POSITIONS: dict[str, tuple[int, ...]] = {
    "furan": (4, 3, 2, 1, 5),
    "thiophene": (4, 3, 2, 1, 5),
    "pyrrole": (4, 3, 2, 1, 5),
    "pyrazole": (4, 3, 2, 1, 5),
    "imidazole": (4, 5, 1, 2, 3),
    "isoxazole": (4, 3, 2, 1, 5),
    "oxazole": (4, 5, 1, 2, 3),
    "isothiazole": (4, 3, 2, 1, 5),
    "thiazole": (4, 5, 1, 2, 3),
    "1,2,3-triazole": (4, 5, 1, 2, 3),
    "1,2,4-triazole": (3, 4, 5, 1, 2),
    "tetrazole": (5, 4, 3, 2, 1),
    "1,2,4-oxadiazole": (3, 4, 5, 1, 2),
    "1,3,4-oxadiazole": (5, 4, 3, 2, 1),
    "1,3,4-thiadiazole": (5, 4, 3, 2, 1),
    "pyridine": (4, 3, 2, 1, 6, 5),
    "pyridazine": (5, 4, 3, 2, 1, 6),
    "pyrimidine": (5, 4, 3, 2, 1, 6),
    "pyrazine": (6, 5, 4, 3, 2, 1),
    "1,3,5-triazine": (6, 5, 4, 3, 2, 1),
    "indole": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "indazole": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "benzimidazole": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "benzofuran": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "benzothiophene": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "benzoxazole": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "benzothiazole": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "7-azaindole": (5, 6, 7, 0, 1, 2, 3, 0, 4),
    "quinoline": (6, 7, 8, 0, 1, 2, 3, 4, 0, 5),
    "isoquinoline": (6, 7, 8, 0, 1, 2, 3, 4, 0, 5),
    "quinazoline": (6, 7, 8, 0, 1, 2, 3, 4, 0, 5),
    "quinoxaline": (6, 7, 8, 0, 1, 2, 3, 4, 0, 5),
    "1,8-naphthyridine": (3, 2, 1, 0, 8, 7, 6, 5, 0, 4),
}

# Ring queries, parsed once, largest first so a fused group numbers as the fused ring.
_NUMBERED_QUERIES = sorted(
    (
        (name, Chem.MolFromSmiles(HETEROCYCLES[name]), positions)
        for name, positions in RING_POSITIONS.items()
    ),
    key=lambda item: -item[1].GetNumAtoms(),
)
_NUMBERED_BY_NAME = {
    name: (query, positions) for name, query, positions in _NUMBERED_QUERIES
}


def _substituents(atom: Chem.Atom, ring: set[int]) -> list[int]:
    """Atomic numbers of the atom's exocyclic neighbours (0 for a port); [] if none."""
    return [n.GetAtomicNum() for n in atom.GetNeighbors() if n.GetIdx() not in ring]


def _match_ring(
    mol: Chem.Mol, query: Chem.Mol, positions: tuple[int, ...]
) -> tuple[dict[int, int], frozenset[int]] | None:
    """Best match of one vendored ring in ``mol``.

    Among the ring's symmetry-equivalent matches, chooses the one giving the
    substituent-bearing atoms the lowest locants (the IUPAC rule), breaking a remaining
    tie by substituent atomic number so the numbering is stable across SMILES forms.

    Args:
        mol: The molecule (or group fragment) to search.
        query: The parsed ring template.
        positions: The template's per-atom IUPAC positions (0 = fusion / unnumbered).

    Returns:
        ``({atom_idx: position}, all_matched_atom_idxs)`` for the chosen match, or None
        when the ring is not a substructure of ``mol``. The atom set keeps the
        position-0 fusion atoms so a caller can tell a ring bond from a substituent.
    """
    matches = mol.GetSubstructMatches(query, uniquify=False)
    if not matches:
        return None
    chosen: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]] | None
    chosen = None
    for match in matches:
        ring = set(match)
        placed = sorted(
            (positions[qi], min(subs))
            for qi, mi in enumerate(match)
            if positions[qi]
            for subs in [_substituents(mol.GetAtomWithIdx(mi), ring)]
            if subs
        )
        # lowest locants first, then lowest substituent atomic numbers by position
        key = (tuple(p for p, _ in placed), tuple(z for _, z in placed))
        if chosen is None or key < chosen[0]:
            chosen = (key, match)
    assert chosen is not None
    numbering = {
        chosen[1][qi]: positions[qi]
        for qi in range(query.GetNumAtoms())
        if positions[qi]
    }
    return numbering, frozenset(chosen[1])


def _best_ring(mol: Chem.Mol) -> tuple[str, dict[int, int], frozenset[int]] | None:
    """The largest vendored ring in ``mol``: (name, numbering, ring atoms), or None."""
    best: tuple[int, str, dict[int, int], frozenset[int]] | None = None
    for name, query, positions in _NUMBERED_QUERIES:
        matched = _match_ring(mol, query, positions)
        if matched is None:
            continue
        size = query.GetNumAtoms()
        if best is None or size > best[0]:
            best = (size, name, matched[0], matched[1])
    return None if best is None else (best[1], best[2], best[3])


def number_ring_system(mol: Chem.Mol) -> tuple[str, dict[int, int]] | None:
    """Assign IUPAC ring positions to a molecule's atoms from a vendored ring.

    Finds the largest vendored ring that is a substructure of ``mol`` and numbers it by
    the IUPAC lowest-locants rule (see ``_match_ring``).

    Args:
        mol: The molecule (or group fragment) to number.

    Returns:
        ``(ring_name, {atom_idx: position})`` for the matched ring, or None when no
        vendored ring matches (a carbocycle, or a ring not in the table).
    """
    ring = _best_ring(mol)
    return None if ring is None else (ring[0], ring[1])


def _render_locants(
    mol: Chem.Mol, name: str, numbering: dict[int, int], ring: frozenset[int]
) -> str:
    """Render a ring's substituent map from a numbering and its full atom set.

    Each numbered position that bears a substituent is listed by IUPAC locant — a port
    as ``[k*] at Cn`` (dummy atom, by isotope), a heavy substituent as its element ``F
    at Cn`` — and each open position (an H-bearing ring atom with no substituent) is
    listed after. Locants use the chemist's ``C2`` / ``N1`` form, not the word
    "position" (which the tools use for the atom-index ``position_id`` of grow/mutate).

    Args:
        mol: The molecule or fragment the numbering came from.
        name: The ring name.
        numbering: ``{atom_idx: position}`` (position-0 fusion atoms omitted).
        ring: All of the ring's atom indices, fusion atoms included.

    Returns:
        e.g. ``"quinazoline (IUPAC numbering): N at C2, C at C6 (open: C4, C5, C7,
        C8)"``.
    """

    def locant(atom_idx: int) -> str:
        return f"{mol.GetAtomWithIdx(atom_idx).GetSymbol()}{numbering[atom_idx]}"

    items: list[tuple[int, str]] = []
    open_atoms: list[tuple[int, int]] = []
    for idx, pos in numbering.items():
        atom = mol.GetAtomWithIdx(idx)
        ext = [n for n in atom.GetNeighbors() if n.GetIdx() not in ring]
        ports = [n for n in ext if n.GetAtomicNum() == 0]
        heavy = [n for n in ext if n.GetAtomicNum() > 1]
        if ports or heavy:
            for port in ports:
                items.append((pos, f"[{port.GetIsotope()}*] at {locant(idx)}"))
            for sub in heavy:
                items.append((pos, f"{sub.GetSymbol()} at {locant(idx)}"))
        elif atom.GetTotalNumHs() > 0 or any(n.GetAtomicNum() == 1 for n in ext):
            open_atoms.append((pos, idx))
    items.sort()
    open_atoms.sort()
    text = f"{name} (IUPAC numbering)"
    if items:
        text += ": " + ", ".join(item for _, item in items)
    if open_atoms:
        text += " (open: " + ", ".join(locant(idx) for _, idx in open_atoms) + ")"
    return text


def ring_locant_summary(mol: Chem.Mol) -> str | None:
    """Name a fragment's largest vendored ring and place every substituent by locant.

    Reports each port and each baked-in heavy substituent by IUPAC locant, plus the open
    positions — so a ring edit can be checked for both ring identity and where each
    substituent sits.

    Args:
        mol: A group fragment (ports are dummy atoms) posed however.

    Returns:
        e.g. ``"quinazoline (IUPAC numbering): [3*] at C2, [4*] at C6 (open: C4, C5,
        C7, C8)"``, or None when the fragment is not a vendored ring.
    """
    ring = _best_ring(mol)
    return None if ring is None else _render_locants(mol, *ring)


def named_ring_locants(mol: Chem.Mol, name: str) -> str | None:
    """Place the substituents of one named ring where it sits in ``mol``.

    Like ``ring_locant_summary`` but pinned to the named ring (not the largest ring
    present), so a match against the ring a request named reports where each substituent
    landed — the check that catches a right ring with a substituent on the wrong carbon.

    Args:
        mol: The whole molecule (or fragment) to number.
        name: A vendored ring name.

    Returns:
        The locant summary, or None when ``name`` is not vendored or the ring is absent.
    """
    entry = _NUMBERED_BY_NAME.get(name)
    if entry is None:
        return None
    matched = _match_ring(mol, *entry)
    return (
        None if matched is None else _render_locants(mol, name, matched[0], matched[1])
    )
