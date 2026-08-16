"""Agent-facing group description: Atom Map, Rings, and Topology renderers."""

from rdkit import Chem

from chemistree.describe import atom_map, rings_section, topology_matrix


def _fragment(smiles: str) -> Chem.Mol:
    """A fragment mol with explicit hydrogens, as the session stores it."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


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


def test_rings_section_benzene_lists_the_ring_atoms():
    mol = _fragment("[1*]c1ccccc1")
    (ring,) = _ring_atom_indices(mol)
    section = rings_section(mol)
    assert "Ring A:" in section
    assert ", ".join(str(i) for i in ring) in section


def test_rings_section_is_empty_for_an_acyclic_fragment():
    assert rings_section(_fragment("[1*]CC")) == ""


def test_rings_section_names_each_ring_of_a_fused_system():
    # Naphthalene: two SSSR rings -> Ring A and Ring B, sharing the fusion atoms.
    mol = _fragment("[1*]c1ccc2ccccc2c1")
    section = rings_section(mol)
    assert "Ring A:" in section
    assert "Ring B:" in section
