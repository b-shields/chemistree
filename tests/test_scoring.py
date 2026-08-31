"""Vinardo empirical pose scoring."""

import pathlib

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from chemistree.scoring import atom_typing, score_intramolecular, score_pose

DATA = pathlib.Path(__file__).parent / "data" / "abl1"


def _carbon_at(x: float, y: float, z: float) -> Chem.Mol:
    """A single-carbon molecule (methane, heavy skeleton) at one point."""
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    heavy = Chem.RemoveAllHs(mol)
    heavy.GetConformer().SetAtomPosition(0, Point3D(x, y, z))
    return heavy


def test_score_of_one_touching_carbon_pair_matches_hand_calculation():
    # Two hydrophobic carbons (radius 2.0) 4.0 A apart -> surface distance 0.
    # gauss1 = 1, hydrophobic = 1, no repulsion, no hbond.
    # score = w_gauss1 * 1 + w_hydrophobic * 1 = -0.045 + -0.035 = -0.080.
    ligand = _carbon_at(0.0, 0.0, 0.0)
    protein = _carbon_at(4.0, 0.0, 0.0)
    typing = atom_typing(protein)
    components = score_pose(ligand, protein.GetConformer().GetPositions(), typing)
    assert components.total == pytest.approx(-0.080, abs=1e-6)


def test_score_falls_off_with_distance():
    # Same pair at 5.0 A: surface distance 1.0.
    # gauss1 = exp(-(1/0.8)^2), hydrophobic = (2.5 - 1.0) / 2.5 = 0.6.
    ligand = _carbon_at(0.0, 0.0, 0.0)
    protein = _carbon_at(5.0, 0.0, 0.0)
    typing = atom_typing(protein)
    components = score_pose(ligand, protein.GetConformer().GetPositions(), typing)
    expected = -0.045 * np.exp(-((1.0 / 0.8) ** 2)) + -0.035 * 0.6
    assert components.total == pytest.approx(expected, abs=1e-6)


def test_repulsion_penalizes_a_clash():
    # Two carbons 2.0 A apart -> surface distance -2.0 -> repulsion = (-2)^2 = 4.
    # The repulsion term (weight +0.8) makes the score strongly positive.
    ligand = _carbon_at(0.0, 0.0, 0.0)
    protein = _carbon_at(2.0, 0.0, 0.0)
    typing = atom_typing(protein)
    components = score_pose(ligand, protein.GetConformer().GetPositions(), typing)
    assert components.repulsion == pytest.approx(0.8 * 4.0, abs=1e-6)
    assert components.total > 0


def test_abl1_crystal_pose_scores_favorably():
    # Ground truth: the co-crystallized ligand pose scores about -11.6 (Vinardo).
    ligand = Chem.MolFromMolFile(str(DATA / "reference.sdf"), removeHs=False)
    receptor = Chem.MolFromPDBFile(
        str(DATA / "receptor.pdb"), removeHs=False, sanitize=False
    )
    heavy = Chem.RemoveAllHs(receptor)
    components = score_pose(
        ligand, heavy.GetConformer().GetPositions(), atom_typing(heavy)
    )
    assert components.total == pytest.approx(-11.6, abs=0.2)


def test_intramolecular_matches_the_validated_cmxflow_value():
    # Ground truth: the reference implementation (cmxflow) scores the crystal
    # ligand's internal Vinardo energy at -0.106673 over 272 non-bonded pairs.
    ligand = Chem.MolFromMolFile(str(DATA / "reference.sdf"), removeHs=False)
    assert score_intramolecular(ligand).total == pytest.approx(-0.106673, abs=1e-4)


def test_intramolecular_is_zero_for_a_single_rigid_ring():
    # Benzene is one rigid fragment, so no pair is in different fragments: nothing
    # is scored, and the internal energy is exactly zero.
    benzene = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(benzene, randomSeed=1)
    assert score_intramolecular(benzene).total == 0.0


def test_intramolecular_penalizes_an_internal_clash():
    # Hexane's terminal carbons are five bonds apart in different rigid fragments,
    # so they are scored; overlapping them drives the internal energy up.
    hexane = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
    AllChem.EmbedMolecule(hexane, randomSeed=1)
    relaxed = score_intramolecular(hexane).total
    conf = hexane.GetConformer()
    heavy = [a.GetIdx() for a in hexane.GetAtoms() if a.GetAtomicNum() > 1]
    conf.SetAtomPosition(heavy[-1], conf.GetAtomPosition(heavy[0]))
    assert score_intramolecular(hexane).total > relaxed
