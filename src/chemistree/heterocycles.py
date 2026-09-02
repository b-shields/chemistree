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


def _substituents(atom: Chem.Atom, ring: set[int]) -> list[int]:
    """Atomic numbers of the atom's exocyclic neighbours (0 for a port); [] if none."""
    return [n.GetAtomicNum() for n in atom.GetNeighbors() if n.GetIdx() not in ring]


def number_ring_system(mol: Chem.Mol) -> tuple[str, dict[int, int]] | None:
    """Assign IUPAC ring positions to a molecule's atoms from a vendored ring.

    Finds the largest vendored ring that is a substructure of ``mol`` and, among that
    ring's symmetry-equivalent matches, chooses the one giving the substituent-bearing
    atoms the lowest locants (the IUPAC rule). A remaining tie on a symmetric ring
    (e.g. pyridine 2 vs 6) is broken by substituent atomic number, so the numbering is
    the same regardless of how the SMILES was written.

    Args:
        mol: The molecule (or group fragment) to number.

    Returns:
        ``(ring_name, {atom_idx: position})`` for the matched ring, or None when no
        vendored ring matches (a carbocycle, or a ring not in the table).
    """
    best: tuple[int, str, dict[int, int]] | None = None
    for name, query, positions in _NUMBERED_QUERIES:
        matches = mol.GetSubstructMatches(query, uniquify=False)
        if not matches:
            continue
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
        size = query.GetNumAtoms()
        if best is None or size > best[0]:
            best = (size, name, numbering)
    return (best[1], best[2]) if best else None


def ring_ports_summary(mol: Chem.Mol) -> str | None:
    """Name a fragment's ring and give each port's IUPAC locant.

    Locants use the chemist's ``C2`` / ``N1`` form (the ring atom's element plus its
    number), not the word "position" — which the tools already use for the atom-index
    ``position_id`` of grow/mutate. The ring is qualified with "IUPAC numbering" once.

    Args:
        mol: A group fragment (ports are dummy atoms) posed however.

    Returns:
        e.g. ``"quinazoline (IUPAC numbering): [3*] at C2, [4*] at C6 (open: C4, C5,
        C7, C8)"``, or None when the fragment is not a vendored ring.
    """
    numbered = number_ring_system(mol)
    if numbered is None:
        return None
    name, positions = numbered

    def locant(atom_idx: int) -> str:
        return f"{mol.GetAtomWithIdx(atom_idx).GetSymbol()}{positions[atom_idx]}"

    ports = sorted(
        (atom.GetIsotope(), atom.GetNeighbors()[0].GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0
        and atom.GetNeighbors()
        and positions.get(atom.GetNeighbors()[0].GetIdx())
    )
    taken = {idx for _, idx in ports}
    open_atoms = sorted(
        (positions[a.GetIdx()], a.GetIdx())
        for a in mol.GetAtoms()
        if positions.get(a.GetIdx())
        and a.GetIdx() not in taken
        and a.GetTotalNumHs() > 0
    )
    text = f"{name} (IUPAC numbering)"
    if ports:
        text += ": " + ", ".join(f"[{label}*] at {locant(idx)}" for label, idx in ports)
    if open_atoms:
        text += " (open: " + ", ".join(locant(idx) for _, idx in open_atoms) + ")"
    return text
