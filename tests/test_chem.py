"""prepare_molecule: explicit-H / 3D normalization across input forms."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from chemistree import prepare_molecule


def _has_explicit_hs(mol: Chem.Mol) -> bool:
    return any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())


def test_prepare_from_smiles_adds_explicit_hs_and_conformer():
    mol = prepare_molecule("Cc1ccccc1")
    assert _has_explicit_hs(mol)
    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().Is3D()


def test_prepare_two_d_is_connectivity_only():
    mol = prepare_molecule("Cc1ccccc1", three_d=False)
    assert not _has_explicit_hs(mol)
    assert mol.GetNumConformers() == 0


def test_prepare_preserves_existing_pose():
    # A loaded 3D pose must be kept; only hydrogens are added.
    posed = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    AllChem.EmbedMolecule(posed, randomSeed=3)
    posed = Chem.RemoveHs(posed)  # heavy-atom 3D pose, implicit Hs

    prepared = prepare_molecule(posed)

    assert _has_explicit_hs(prepared)
    before = posed.GetConformer()
    match = prepared.GetSubstructMatch(posed)
    after = prepared.GetConformer()
    for old_idx, new_idx in enumerate(match):
        assert tuple(before.GetAtomPosition(old_idx)) == pytest.approx(
            tuple(after.GetAtomPosition(new_idx)), abs=1e-6
        )


def test_prepare_rejects_unparseable_smiles():
    with pytest.raises(ValueError):
        prepare_molecule("not a molecule")
