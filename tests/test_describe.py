"""Agent-facing group description: Atom Map, Rings, Topology, and Positions."""

import re

from rdkit import Chem

from chemistree.describe import atom_map, positions, rings_section, topology_matrix


def _fragment(smiles: str) -> Chem.Mol:
    """A fragment mol with explicit hydrogens, as the session stores it."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _positions_map(block: str) -> dict[int, dict[str, list[int]]]:
    """Parse a positions block into ``{atom_id: {label: [neighbor ids]}}``.

    Each neighbor is an ``[elem:id]`` token, optionally carrying a ``·[n*]`` port
    annotation; the ``id`` is extracted so tests assert relationships without
    depending on the element or annotation text.
    """
    out: dict[int, dict[str, list[int]]] = {}
    for line in block.splitlines():
        if not line.startswith("- atom "):
            continue
        head, _, rels = line.partition(" — ")
        atom_id = int(head.split()[2])
        labels: dict[str, list[int]] = {}
        if rels:
            for part in rels.split(";"):
                label, _, ids = part.strip().partition(":")
                found = [re.search(r"\[[A-Za-z]+:(\d+)\]", t) for t in ids.split(",")]
                labels[label.strip()] = [int(m.group(1)) for m in found if m]
        out[atom_id] = labels
    return out


def _descriptor(block: str, atom_id: int) -> str:
    """The parenthesized descriptor of one atom's positions line."""
    line = next(ln for ln in block.splitlines() if ln.startswith(f"- atom {atom_id} ("))
    return line[line.index("(") + 1 : line.index(")")]


def _parse_keeping_hs(smiles: str) -> Chem.Mol:
    """Parse a SMILES without folding explicit hydrogens into their neighbors."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)


def _ring_atom_indices(mol: Chem.Mol) -> list[list[int]]:
    """SSSR rings as sorted heavy-atom index lists, for ground truth."""
    return [sorted(ring) for ring in mol.GetRingInfo().AtomRings()]


def _matrix_rows(block: str) -> dict[str, dict[str, str]]:
    """Parse a rendered topology block into ``{row_label: {col_label: cell}}``.

    Robust to column padding: tokens are split on whitespace, so tests assert
    cell values without depending on the exact spacing.
    """
    lines = [line for line in block.splitlines() if line.strip()]
    header = lines[0].split()
    rows = {}
    for line in lines[1:]:
        tokens = line.split()
        label, cells = tokens[0], tokens[1:]
        rows[label] = dict(zip(header, cells))
    return rows


def test_atom_map_labels_every_atom_with_its_rdkit_index():
    mol = _fragment("[1*]CC")
    parsed = _parse_keeping_hs(atom_map(mol))
    # Every heavy/H atom carries its own rdkit index as the atom-map number.
    labeled = {a.GetAtomMapNum(): a.GetSymbol() for a in parsed.GetAtoms()}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            assert labeled[atom.GetIdx()] == atom.GetSymbol()


def test_atom_map_renders_index_zero():
    # When a heavy atom sits at rdkit index 0 (as in real, heavy-first fragments)
    # it must be visibly labeled `:0` — the case plain SMILES atom-map notation
    # cannot emit without the +1/-1 trick.
    mol = _fragment("C[1*]")  # carbon at index 0, port at index 1
    assert ":0" in atom_map(mol)


def test_atom_map_keeps_ports_as_labels_not_atom_ids():
    # A port renders as `[n*]` (its pairing label), never as a `:k` atom id.
    mol = _fragment("[2*]CC")
    mapped = atom_map(mol)
    assert "[2*]" in mapped
    assert "*:" not in mapped  # the dummy never gets an atom-map id


def test_topology_matrix_benzene_ring_distances():
    # Ground truth on a benzene ring: ortho=1, meta=2, para=3.
    mol = _fragment("[1*]c1ccccc1")
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    rows = _matrix_rows(topology_matrix(mol))
    c0 = str(carbons[0])
    assert rows[c0][str(carbons[0])] == "."  # diagonal
    assert rows[c0][str(carbons[1])] == "1"  # ortho
    assert rows[c0][str(carbons[2])] == "2"  # meta
    assert rows[c0][str(carbons[3])] == "3"  # para


def test_topology_matrix_has_no_hydrogen_rows():
    # Only heavy atoms and ports get rows/cols; hydrogens are excluded.
    mol = _fragment("[1*]c1ccccc1")
    rows = _matrix_rows(topology_matrix(mol))
    hydrogens = {str(a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() == 1}
    assert not (hydrogens & set(rows))


def test_topology_matrix_includes_the_port_column():
    mol = _fragment("[1*]c1ccccc1")
    rows = _matrix_rows(topology_matrix(mol))
    assert "[1*]" in next(iter(rows.values()))


def test_rings_section_benzene_lists_size_and_ring_atoms():
    mol = _fragment("[1*]c1ccccc1")
    (ring,) = _ring_atom_indices(mol)
    section = rings_section(mol)
    assert "Ring A (6-membered):" in section  # size stated so the agent needn't count
    assert ", ".join(str(i) for i in ring) in section


def test_rings_section_is_empty_for_an_acyclic_fragment():
    assert rings_section(_fragment("[1*]CC")) == ""


def test_rings_section_names_each_ring_of_a_fused_system():
    # Naphthalene: two SSSR rings -> Ring A and Ring B, sharing the fusion atoms.
    mol = _fragment("[1*]c1ccc2ccccc2c1")
    section = rings_section(mol)
    assert "Ring A (6-membered):" in section
    assert "Ring B (6-membered):" in section


def _distances_from(mol: Chem.Mol, source: int, distance: int) -> list[int]:
    """Heavy-atom ids exactly ``distance`` bonds from ``source``."""
    dmat = Chem.GetDistanceMatrix(mol)
    return sorted(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1 and int(dmat[source][a.GetIdx()]) == distance
    )


def test_positions_aromatic_uses_ortho_meta_para():
    mol = _fragment("[1*]c1ccccc1")
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    x = carbons[0]
    labels = _positions_map(positions(mol))[x]
    assert labels["ortho"] == _distances_from(mol, x, 1)
    assert labels["meta"] == _distances_from(mol, x, 2)
    assert labels["para"] == _distances_from(mol, x, 3)


def test_positions_aliphatic_uses_bond_counts():
    mol = _fragment("[1*]C1CCCCC1")  # cyclohexane: non-aromatic ring
    carbon = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    labels = _positions_map(positions(mol))[carbon]
    assert labels["1 bond"] == _distances_from(mol, carbon, 1)
    assert labels["2 bonds"] == _distances_from(mol, carbon, 2)
    assert labels["3 bonds"] == _distances_from(mol, carbon, 3)
    assert "ortho" not in labels  # aromatic terms never used off an aromatic ring
    assert "alpha" not in labels  # greek is gone


def test_positions_five_membered_ring_uses_bond_counts():
    # A 5-membered aromatic ring is not benzene: ortho/meta/para never apply there;
    # its neighbours are given as plain bond counts.
    mol = _fragment("c1ccsc1")  # thiophene
    carbon = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C")
    labels = _positions_map(positions(mol))[carbon]
    assert labels["1 bond"] == _distances_from(mol, carbon, 1)
    assert labels["2 bonds"] == _distances_from(mol, carbon, 2)
    assert "ortho" not in labels
    assert "meta" not in labels


def test_positions_fused_ring_uses_bond_counts():
    # A fused 6,6 system (naphthalene) is not a benzene: chemists number it, so
    # ortho/meta/para never apply -- every relation is a plain bond count.
    mol = _fragment("c1ccc2ccccc2c1")  # naphthalene
    block = positions(mol)
    assert "ortho" not in block
    assert "meta" not in block
    assert "para" not in block
    carbon = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C")
    labels = _positions_map(block)[carbon]
    assert labels["1 bond"] == _distances_from(mol, carbon, 1)


def test_positions_lists_hydrogen_ids_for_grow():
    mol = _fragment("[1*]c1ccccc1")
    carbon = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetTotalNumHs(includeNeighbors=True) == 1
    )
    hydrogen = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(carbon).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    assert f"H {hydrogen}" in _descriptor(positions(mol), carbon)


def test_positions_annotates_ports_on_bearer_and_as_landmark():
    mol = _fragment("[1*]c1ccccc1")
    bearer = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if any(n.GetAtomicNum() == 0 for n in a.GetNeighbors())
    )
    block = positions(mol)
    assert "bears [1*]" in _descriptor(block, bearer)  # the site itself
    # An ortho neighbour of the bearer lists it with the port annotation.
    neighbor = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(bearer).GetNeighbors()
        if n.GetAtomicNum() > 1
    )
    line = next(ln for ln in block.splitlines() if ln.startswith(f"- atom {neighbor} "))
    assert f":{bearer}]·[1*]" in line  # e.g. [c:0]·[1*]


def test_positions_radius_limits_the_range():
    mol = _fragment("[1*]c1ccccc1")
    x = next(a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic())
    labels = _positions_map(positions(mol, radius=1))[x]
    assert labels["ortho"] == _distances_from(mol, x, 1)
    assert "meta" not in labels  # radius 1 stops at ortho
    assert "para" not in labels
