"""Receptor proximity resolution: synthetic units and the ABL1 docked pose."""

import pathlib

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from chemistree import DesignSession
from chemistree.errors import NotFound
from chemistree.fragment import Fragment
from chemistree.receptor import Receptor, min_distance
from chemistree.tree import FragmentNode

DATA = pathlib.Path(__file__).parent / "data" / "abl1"


def _receptor(specs) -> Receptor:
    """A minimal receptor: one atom per residue at a given position."""
    rw = Chem.RWMol()
    for name, number, _ in specs:
        atom = Chem.Atom(6)
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(name)
        info.SetResidueNumber(number)
        info.SetChainId("A")
        atom.SetMonomerInfo(info)
        rw.AddAtom(atom)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (_, _, pos) in enumerate(specs):
        conf.SetAtomPosition(i, Point3D(*pos))
    mol.AddConformer(conf)
    return Receptor(mol)


def _methyl_node(pos, node_id: int) -> FragmentNode:
    """A methyl fragment whose carbon sits at ``pos``."""
    mol = Chem.AddHs(Chem.MolFromSmiles("[*]C"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    conf = mol.GetConformer()
    carbon = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    shift = np.array(pos) - np.array(list(conf.GetAtomPosition(carbon)))
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(
            i, Point3D(*(np.array(list(conf.GetAtomPosition(i))) + shift))
        )
    node = FragmentNode(Fragment(mol))
    node.id = node_id
    return node


def test_min_distance():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
    assert min_distance(a, b) == 5.0


def test_residues_group_and_filter():
    receptor = _receptor(
        [("PHE", 1, (0, 0, 0)), ("ALA", 2, (5, 0, 0)), ("PHE", 3, (9, 0, 0))]
    )
    assert len(receptor.residues()) == 3
    phe = receptor.residues("PHE")
    assert [r.number for r in phe] == [1, 3]


def test_nearest_returns_the_closest_candidate():
    receptor = _receptor([("PHE", 1, (0, 0, 0))])
    near = _methyl_node((2, 0, 0), node_id=0)
    far = _methyl_node((7, 0, 0), node_id=1)
    assert receptor.nearest([near, far], "PHE").id == 0


def test_nearest_raises_beyond_the_proximity_radius():
    receptor = _receptor([("PHE", 1, (0, 0, 0))])
    far = _methyl_node((20, 0, 0), node_id=0)
    with pytest.raises(NotFound):
        receptor.nearest([far], "PHE")


def _abl1_session() -> DesignSession:
    ligand = Chem.MolFromMolFile(str(DATA / "reference.sdf"), removeHs=False)
    receptor = Chem.MolFromPDBFile(
        str(DATA / "receptor.pdb"), removeHs=False, sanitize=False
    )
    return DesignSession(ligand, receptor)


def test_pocket_residues_line_the_binding_site():
    session = _abl1_session()
    pocket = session.receptor.pocket(session.molecule())
    assert len(pocket) > 5
    assert "ASP" in {residue.name for residue in pocket}  # a known contact


def test_residues_match_name_and_optional_number():
    # A chemist says "ALA37"; a bare "ALA" still matches every alanine.
    receptor = _receptor(
        [("ALA", 37, (0, 0, 0)), ("ALA", 99, (5, 0, 0)), ("PHE", 40, (9, 0, 0))]
    )
    assert len(receptor.residues("ALA")) == 2
    numbered = receptor.residues("ALA37")
    assert [(r.name, r.number) for r in numbered] == [("ALA", 37)]
    assert receptor.residues("ALA100") == []
