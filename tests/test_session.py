"""The redesigned, id-addressed design session."""

import itertools
import pathlib
import re
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chemistree.fragment import Fragment
from chemistree.fragmenter import fragment
from chemistree.session import DesignSession

DATA = pathlib.Path(__file__).parent / "data" / "abl1"


def _abl1_session() -> DesignSession:
    """A posed abl1 ligand with its receptor, for spatial tests."""
    ligand = Chem.MolFromMolFile(str(DATA / "reference.sdf"), removeHs=False)
    receptor = Chem.MolFromPDBFile(
        str(DATA / "receptor.pdb"), removeHs=False, sanitize=False
    )
    return DesignSession(ligand, receptor)


def _fragment_multiset(mol_or_tree) -> Counter:
    """Canonical fragment SMILES (ports as bare `*`) of a tree or a fresh mol."""
    tree = mol_or_tree if hasattr(mol_or_tree, "nodes") else fragment(mol_or_tree)
    smiles = []
    for node in tree.nodes:
        m = Chem.Mol(node.current.mol)
        for atom in m.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)  # ignore port labels; compare shapes only
        smiles.append(Chem.MolToSmiles(Chem.RemoveHs(m)))
    return Counter(smiles)


def _built_from(session) -> Counter:
    """The fragment multiset of constructing fresh from the current molecule."""
    return _fragment_multiset(session.molecule())


def test_describe_is_a_compact_overview_without_per_group_detail():
    session = DesignSession("Cc1ccc(N)cc1", three_d=False)
    text = session.describe()
    assert text.startswith("# Group Summary")
    assert text.count("\n- [") == len(session.tree.nodes)  # one bullet per group
    # The heavy per-group detail is pulled on demand, not dumped here.
    assert "**Atom Map:**" not in text
    assert "**Topology:**" not in text


def test_describe_group_shows_atom_map_topology_and_rings():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    methyl = next(
        n.id
        for n in session.tree.nodes
        if sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == 1
    )
    ring_detail = session.describe_group(ring)
    assert "**Atom Map:**" in ring_detail
    assert "**Positions:**" in ring_detail
    assert "**Rings:**" in ring_detail
    assert "position_id" in ring_detail  # the legend lives with the detail
    assert "**Rings:**" not in session.describe_group(methyl)  # acyclic
    # The raw matrix is available behind the flag for comparison.
    assert "**Topology:**" in session.describe_group(ring, use_matrix=True)


def test_smiles_history_records_construction():
    session = DesignSession("Cc1ccccc1", three_d=False)
    assert session.smiles_history == [Chem.CanonSmiles("Cc1ccccc1")]


def _aromatic_h_on_ring(session, node_id):
    """An id of a hydrogen on the aromatic ring of the given node."""
    mol = session.tree.node(node_id).current.mol
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetIsAromatic():
            return atom.GetIdx()
    raise AssertionError("no aromatic hydrogen found")


def _methyl_h(session, node_id):
    """An id of a hydrogen on a (single-carbon) methyl node."""
    mol = session.tree.node(node_id).current.mol
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)


def test_swap_obeys_the_edit_construct_invariant():
    session = DesignSession("Cc1ccccc1", three_d=False)
    # swap the methyl leaf for a benzyl group, which itself splits at construction
    leaf = next(
        n.id
        for n in session.tree.nodes
        if sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == 1
    )
    session.swap(leaf, "[*]Cc1ccccc1")
    assert _fragment_multiset(session.tree) == _built_from(session)


def test_grow_split_case_adds_a_new_node_on_a_ring():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    before = len(session.tree.nodes)
    grown = session.grow(ring, _aromatic_h_on_ring(session, ring), "methyl")
    assert session.smiles() == Chem.CanonSmiles("Cc1ccccc1C")  # an o-/m-/p-xylene
    assert len(session.tree.nodes) == before + 1  # a new leaf node
    assert (
        grown not in {n for n in range(before)} or grown == session.tree.node(grown).id
    )
    assert _fragment_multiset(session.tree) == _built_from(session)


def test_grow_merge_case_extends_the_fragment_without_a_new_node():
    session = DesignSession("Cc1ccccc1", three_d=False)
    methyl = next(
        n.id
        for n in session.tree.nodes
        if sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == 1
    )
    before = len(session.tree.nodes)
    session.grow(methyl, _methyl_h(session, methyl), "methyl")
    assert session.smiles() == Chem.CanonSmiles("CCc1ccccc1")  # ethylbenzene
    assert len(session.tree.nodes) == before  # merged: no new node
    assert _fragment_multiset(session.tree) == _built_from(session)


def test_grow_rejects_a_heavy_atom_position():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    with pytest.raises(ValueError, match="hydrogen"):
        session.grow(ring, 0, "methyl")  # atom 0 is a heavy ring carbon


def test_mutate_ring_carbon_to_nitrogen():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    # a ring carbon that bears a hydrogen (not the one carrying the methyl port)
    mol = session.tree.node(ring).current.mol
    carbon = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetTotalNumHs(includeNeighbors=True) == 1
    )
    session.mutate(ring, carbon, "N")
    assert session.smiles() == Chem.CanonSmiles("Cc1ccccn1")  # a methylpyridine


def test_mutate_rejects_a_hydrogen_position():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    with pytest.raises(ValueError, match="heavy atom"):
        session.mutate(ring, _aromatic_h_on_ring(session, ring), "N")


def test_mutate_invalid_element_raises_descriptive_error():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    mol = session.tree.node(ring).current.mol
    carbon = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetTotalNumHs(includeNeighbors=True) == 1
    )
    with pytest.raises(ValueError, match="aromatic"):
        session.mutate(ring, carbon, "O")  # oxygen cannot hold the aromatic ring


def test_grow_in_3d_keeps_bond_lengths_sane():
    # A warped placement shows up as an over-long heavy bond or a collapsed C-H.
    session = DesignSession("Cc1ccccc1", three_d=True)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    session.grow(ring, _aromatic_h_on_ring(session, ring), "ethyl")
    conf = session.molecule().GetConformer()
    heavy, ch = [], []
    for bond in session.molecule().GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        d = conf.GetAtomPosition(a.GetIdx()).Distance(conf.GetAtomPosition(b.GetIdx()))
        if a.GetAtomicNum() > 1 and b.GetAtomicNum() > 1:
            heavy.append(d)
        if {a.GetAtomicNum(), b.GetAtomicNum()} == {6, 1}:
            ch.append(d)
    assert max(heavy) < 1.7
    assert min(ch) > 1.0


def test_grow_on_a_substituted_ring_keeps_the_ring_intact():
    # Regression: growing on a ring that already carries substituents must not
    # re-embed and warp the scaffold. The bug pretzeled the ring, stretching its
    # bonds to ~2.4 A and the new bond to ~4.2 A.
    session = DesignSession("Clc1ccccc1", three_d=True)  # chlorobenzene
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    before = _ring_bond_lengths(session.tree.node(ring).current.mol)
    session.grow(ring, _aromatic_h_on_ring(session, ring), "[*]CO")  # methyl alcohol

    mol = session.molecule()
    conf = mol.GetConformer()
    heavy = [
        conf.GetAtomPosition(b.GetBeginAtomIdx()).Distance(
            conf.GetAtomPosition(b.GetEndAtomIdx())
        )
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1
    ]
    assert max(heavy) < 1.9  # no over-long bond anywhere (C-Cl is ~1.74)
    # The aromatic ring keeps the geometry it had before the grow.
    after = _ring_bond_lengths(session.tree.node(ring).current.mol)
    assert after == pytest.approx(before, abs=1e-3)


def test_swap_reshaping_a_carbon_keeps_it_tetrahedral():
    # Regression: swapping a methyl (-CH3) for a hydroxymethyl (-CH2OH) changes
    # the carbon's substituents (an H gives way to an O). The overlay must not
    # pin the surviving hydrogens onto the old three-hydrogen frame; doing so left
    # no tetrahedral slot for the new oxygen and folded it in to a ~58 deg angle.
    session = DesignSession("Cc1ccccc1", three_d=True)  # toluene
    methyl = next(
        n.id
        for n in session.tree.nodes
        if sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == 1
    )
    session.swap(methyl, "[*]CO")  # -> benzyl alcohol
    assert session.smiles() == Chem.CanonSmiles("OCc1ccccc1")

    mol = session.molecule()
    conf = mol.GetConformer()
    site = next(
        a
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and any(n.GetAtomicNum() == 8 for n in a.GetNeighbors())
        and any(n.GetIsAromatic() for n in a.GetNeighbors())
    )
    p = np.array(conf.GetAtomPosition(site.GetIdx()))
    neighbors = [n.GetIdx() for n in site.GetNeighbors()]
    for i, j in itertools.combinations(neighbors, 2):
        vi = np.array(conf.GetAtomPosition(i)) - p
        vj = np.array(conf.GetAtomPosition(j)) - p
        vi /= np.linalg.norm(vi)
        vj /= np.linalg.norm(vj)
        angle = np.degrees(np.arccos(np.clip(np.dot(vi, vj), -1, 1)))
        assert angle > 100  # every angle tetrahedral, not a folded ~58 deg


def _ring_bond_lengths(mol: Chem.Mol) -> list[float]:
    """Sorted lengths of the aromatic ring bonds, for a warp check."""
    conf = mol.GetConformer()
    return sorted(
        conf.GetAtomPosition(b.GetBeginAtomIdx()).Distance(
            conf.GetAtomPosition(b.GetEndAtomIdx())
        )
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic()
    )


def test_2d_only_grow_and_mutate_need_no_conformer():
    session = DesignSession("Cc1ccccc1", three_d=False)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    session.grow(ring, _aromatic_h_on_ring(session, ring), "[*]F")
    assert session.smiles() == Chem.CanonSmiles("Cc1ccccc1F")
    assert "**Atom Map:**" in session.describe_group(ring)


def test_distance_report_matches_ground_truth_minimum():
    session = _abl1_session()
    report = session.distance("ASP")
    assert report.startswith("# Distances to ASP")
    assert "## Details" in report

    # The smallest distance printed must equal the true closest ligand-ASP approach.
    residue = session.receptor.residue_atoms("ASP")
    truth = min(
        float(
            np.sqrt(
                ((residue - np.array(list(conf.GetAtomPosition(a.GetIdx())))) ** 2).sum(
                    -1
                )
            ).min()
        )
        for node in session.tree.nodes
        for conf in [node.current.mol.GetConformer()]
        for a in node.current.mol.GetAtoms()
        if a.GetAtomicNum() > 1
    )
    printed = min(
        float(token)
        for line in report.splitlines()
        if line.startswith("| [")
        for token in [line.split("|")[2].strip()]
    )
    assert printed == pytest.approx(truth, abs=0.01)


def test_distance_requires_a_receptor():
    session = DesignSession("Cc1ccccc1", three_d=True)
    with pytest.raises(ValueError, match="receptor"):
        session.distance("ASP")


def test_contacts_lists_the_closest_group_and_atom_per_site_residue():
    session = _abl1_session()
    report = session.contacts(dist_cutoff=4.5)
    assert report.startswith("# Binding-site contacts (within 4.5 A)")

    # One row per pocket residue; a known contact residue appears.
    pocket = session.receptor.pocket(session.molecule(), within=4.5)
    rows = [line for line in report.splitlines() if " | atom " in line]
    assert len(rows) == len(pocket)
    assert any("ASP" in row for row in rows)


def test_contacts_distance_matches_ground_truth_for_a_residue():
    session = _abl1_session()
    report = session.contacts(dist_cutoff=4.5)

    # Pick a residue named in the report and confirm its printed distance equals
    # the true closest approach of any (port-excluded) ligand heavy atom.
    pocket = session.receptor.pocket(session.molecule(), within=4.5)
    residue = pocket[0]
    spec = f"{residue.name}{residue.number}"
    coords = session.receptor.residue_positions(residue)
    truth = min(
        float(
            np.sqrt(
                ((coords - np.array(list(conf.GetAtomPosition(a.GetIdx())))) ** 2).sum(
                    -1
                )
            ).min()
        )
        for node in session.tree.nodes
        for conf in [node.current.mol.GetConformer()]
        for a in node.current.mol.GetAtoms()
        if a.GetAtomicNum() > 1
    )
    row = next(line for line in report.splitlines() if f"| {spec} " in line)
    printed = float(row.split("|")[4].strip())
    assert printed == pytest.approx(truth, abs=0.01)


def test_contacts_never_cites_a_port_atom():
    session = _abl1_session()
    report = session.contacts(dist_cutoff=6.0)
    for line in report.splitlines():
        if " | atom " not in line:
            continue
        # Row: | ResName | [id] label | atom N (Sym) | dist |
        node_id = int(line.split("]")[0].split("[")[1])
        atom_id = int(line.split("atom ")[1].split(" ")[0])
        atom = session.tree.node(node_id).current.mol.GetAtomWithIdx(atom_id)
        assert atom.GetAtomicNum() > 1  # never a hydrogen or a dummy port


def test_contacts_requires_a_receptor():
    session = DesignSession("Cc1ccccc1", three_d=True)
    with pytest.raises(ValueError, match="receptor"):
        session.contacts()


def test_edits_are_undoable():
    session = DesignSession("Cc1ccccc1", three_d=False)
    start = session.smiles()
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    session.grow(ring, _aromatic_h_on_ring(session, ring), "methyl")
    session.undo()
    assert session.smiles() == start


def _positions(mol: Chem.Mol) -> np.ndarray:
    """Every atom's coordinates as an (N, 3) array."""
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _leaf_by_heavy(session: DesignSession, n_heavy: int) -> int:
    """Id of the single-port leaf with a given heavy-atom count."""
    node = next(
        n
        for n in session.tree.nodes
        if len(n.current.ports) == 1
        and sum(1 for a in n.current.mol.GetAtoms() if a.GetAtomicNum() > 1) == n_heavy
    )
    assert node.id is not None
    return node.id


def test_rotate_leaves_the_constitution_unchanged():
    session = DesignSession("CCc1ccccc1", three_d=True)  # ethylbenzene
    before = session.smiles()
    session.rotate(_leaf_by_heavy(session, 2), 120)  # the ethyl
    assert session.smiles() == before  # a torsion changes coordinates, not the graph


def test_rotate_turns_the_group_and_leaves_the_scaffold_fixed():
    session = DesignSession("CCc1ccccc1", three_d=True)
    ethyl = _leaf_by_heavy(session, 2)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    ring_before = _positions(session.tree.node(ring).current.mol)
    ethyl_before = _positions(session.tree.node(ethyl).current.mol)
    session.rotate(ethyl, 120)
    # The ring (the other side of the bond) does not move.
    assert np.allclose(ring_before, _positions(session.tree.node(ring).current.mol))
    # The ethyl does move.
    assert not np.allclose(
        ethyl_before, _positions(session.tree.node(ethyl).current.mol)
    )


def test_rotate_carries_a_ring_substituent_rigidly():
    # A substituted ring hanging off a scaffold: rotating the ring about the bond
    # to the scaffold must carry its substituent, as one rigid body.
    session = DesignSession("c1ccccc1-c1ccc(C)cc1", three_d=True)  # 4-methylbiphenyl

    # The substituted ring is the six-membered ring bearing the methyl (2 edges).
    ring_ids = [
        n.id
        for n in session.tree.nodes
        if n.current.mol.GetRingInfo().NumRings() > 0 and n.id is not None
    ]
    sub_ring = next(
        nid
        for nid in ring_ids
        if len(session.tree.neighbors(session.tree.node(nid))) == 2
    )
    plain_ring = next(nid for nid in ring_ids if nid != sub_ring)
    methyl = _leaf_by_heavy(session, 1)

    plain_before = _positions(session.tree.node(plain_ring).current.mol)
    ring_before = _positions(session.tree.node(sub_ring).current.mol)
    methyl_c_before = _positions(session.tree.node(methyl).current.mol)
    # distance between a ring atom and the methyl carbon, to check rigidity
    gap_before = float(
        np.linalg.norm(
            ring_before[0] - methyl_c_before[_methyl_carbon(session, methyl)]
        )
    )

    session.rotate(sub_ring, 90)

    # The plain ring (scaffold side) is fixed; the substituted ring turned.
    assert np.allclose(
        plain_before, _positions(session.tree.node(plain_ring).current.mol)
    )
    assert not np.allclose(
        ring_before, _positions(session.tree.node(sub_ring).current.mol)
    )
    # The methyl moved with the ring, preserving their separation (rigid subtree).
    ring_after = _positions(session.tree.node(sub_ring).current.mol)
    methyl_c_after = _positions(session.tree.node(methyl).current.mol)
    assert not np.allclose(methyl_c_before, methyl_c_after)  # the methyl came along
    gap_after = float(
        np.linalg.norm(ring_after[0] - methyl_c_after[_methyl_carbon(session, methyl)])
    )
    assert gap_after == pytest.approx(gap_before, abs=1e-6)


def _methyl_carbon(session: DesignSession, methyl_id: int) -> int:
    """Index of the carbon in a methyl fragment."""
    mol = session.tree.node(methyl_id).current.mol
    return int(next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6))


def test_rotate_is_undoable():
    session = DesignSession("CCc1ccccc1", three_d=True)
    ethyl = _leaf_by_heavy(session, 2)
    before = _positions(session.tree.node(ethyl).current.mol)
    session.rotate(ethyl, 120)
    session.undo()
    assert np.allclose(before, _positions(session.tree.node(ethyl).current.mol))


def test_rotate_requires_3d_coordinates():
    session = DesignSession("CCc1ccccc1", three_d=False)
    with pytest.raises(ValueError, match="3D"):
        session.rotate(_leaf_by_heavy(session, 2), 120)


def test_clashes_reports_none_for_a_clean_structure():
    session = DesignSession("Cc1ccccc1", three_d=True)  # a relaxed toluene
    assert "No clashes." in session.clashes()


def test_clashes_names_the_two_overlapping_groups():
    # Force the ethyl's terminal carbon onto a ring carbon, then detect the clash.
    session = DesignSession("CCc1ccccc1", three_d=True)
    ethyl = _leaf_by_heavy(session, 2)
    ring = next(
        n.id for n in session.tree.nodes if n.current.mol.GetRingInfo().NumRings()
    )
    ring_carbon = _positions(session.tree.node(ring).current.mol)[0]

    node = session.tree.node(ethyl)
    mol = Chem.Mol(node.current.mol)
    conf = mol.GetConformer()
    anchor = node.current.ports[0].anchor_idx
    terminal = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetIdx() != anchor
    )
    shift = ring_carbon - np.array(list(conf.GetAtomPosition(terminal)))
    for i in range(mol.GetNumAtoms()):
        p = np.array(list(conf.GetAtomPosition(i))) + shift
        conf.SetAtomPosition(i, Point3D(*p))
    node.push(Fragment(mol))

    report = session.clashes()
    assert "No clashes." not in report
    assert f"[{ethyl}]" in report and f"[{ring}]" in report


def test_clashes_requires_3d_coordinates():
    session = DesignSession("Cc1ccccc1", three_d=False)
    with pytest.raises(ValueError, match="3D"):
        session.clashes()


def test_describe_shows_predicted_affinity_for_the_crystal_pose():
    session = _abl1_session()
    describe = session.describe()
    assert "Predicted affinity (Vinardo):" in describe
    # The co-crystallized pose scores about -11.6 (matches the scoring unit test).
    assert _affinity(describe) == pytest.approx(-11.6, abs=0.2)


def test_describe_affinity_worsens_when_a_group_is_shoved_into_the_receptor():
    session = _abl1_session()
    before = _affinity(session.describe())
    # Shove one group into the receptor: the clash penalty must raise the score.
    node = session.tree.nodes[0]
    mol = Chem.Mol(node.current.mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = np.array(list(conf.GetAtomPosition(i))) + np.array([1.5, 0.0, 0.0])
        conf.SetAtomPosition(i, Point3D(*p))
    node.push(Fragment(mol))
    assert _affinity(session.describe()) > before


def test_describe_omits_affinity_without_a_receptor():
    session = DesignSession("Cc1ccccc1", three_d=True)
    assert "Predicted affinity" not in session.describe()


def test_describe_omits_affinity_without_3d_coordinates():
    receptor = Chem.MolFromPDBFile(
        str(DATA / "receptor.pdb"), removeHs=False, sanitize=False
    )
    session = DesignSession("Cc1ccccc1", receptor, three_d=False)
    assert "Predicted affinity" not in session.describe()


def _affinity(describe: str) -> float:
    """The numeric Vinardo affinity parsed from a describe overview."""
    line = next(ln for ln in describe.splitlines() if "Predicted affinity" in ln)
    match = re.search(r"(-?\d+\.\d+)", line)
    assert match is not None
    return float(match.group(1))
