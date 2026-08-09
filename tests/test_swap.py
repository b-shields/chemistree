"""Swap edit: replacing a fragment while preserving ports and 3D placement."""

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemistree import fragment, prepare_molecule, swap


def _embed(smiles: str) -> Chem.Mol:
    """A molecule with explicit Hs and a 3D conformer (the 3D edit path)."""
    return prepare_molecule(smiles)


def _heavy(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _leaf(tree, n_heavy: int):
    return next(
        node
        for node in tree.nodes
        if _heavy(node.current.mol) == n_heavy and len(node.current.ports) == 1
    )


def _ring_node(tree):
    return next(
        node
        for node in tree.nodes
        if any(a.GetIsAromatic() for a in node.current.mol.GetAtoms())
    )


def _product_smiles(tree) -> str:
    return str(Chem.MolToSmiles(Chem.RemoveHs(tree.reconstruct())))


def _aromatic_coords(mol: Chem.Mol):
    conf = mol.GetConformer()
    return sorted(
        tuple(round(c, 4) for c in conf.GetAtomPosition(a.GetIdx()))
        for a in mol.GetAtoms()
        if a.GetIsAromatic()
    )


def test_swap_single_atom_methyl_to_cf3():
    tree = fragment(_embed("Cc1ccccc1"))
    swap(_leaf(tree, 1), "[*]C(F)(F)F")
    assert _product_smiles(tree) == Chem.CanonSmiles("FC(F)(F)c1ccccc1")


def test_swap_single_atom_methyl_to_fluorine():
    tree = fragment(_embed("Cc1ccccc1"))
    swap(_leaf(tree, 1), "[*]F")
    assert _product_smiles(tree) == Chem.CanonSmiles("Fc1ccccc1")


def test_swap_group_ethyl_to_fluoroethyl_keeps_core():
    tree = fragment(_embed("CCc1ccccc1"))
    ethyl = _leaf(tree, 2)
    anchor_before = tuple(
        ethyl.current.mol.GetConformer().GetAtomPosition(
            ethyl.current.ports[0].anchor_idx
        )
    )
    swap(ethyl, "[*]CCF")
    assert _product_smiles(tree) == Chem.CanonSmiles("FCCc1ccccc1")
    anchor_after = tuple(
        ethyl.current.mol.GetConformer().GetAtomPosition(
            ethyl.current.ports[0].anchor_idx
        )
    )
    assert anchor_before == pytest.approx(anchor_after, abs=1e-6)


def test_swap_ring_benzene_to_pyridine():
    tree = fragment(_embed("Cc1ccc(C)cc1"))  # p-xylene
    ring = _ring_node(tree)
    a, b = (p.label for p in ring.current.ports)
    swap(ring, f"[{a}*]c1ccc([{b}*])nc1")
    product = Chem.RemoveHs(tree.reconstruct())
    assert rdMolDescriptors.CalcMolFormula(product) == "C7H9N"


def test_swap_ring_benzene_to_oxazole_contraction():
    tree = fragment(_embed("Cc1ccccc1"))  # toluene, single-port ring
    swap(_ring_node(tree), "[*]c1ocnc1")
    product = Chem.RemoveHs(tree.reconstruct())
    assert rdMolDescriptors.CalcMolFormula(product) == "C4H5NO"


def test_grow_via_swap_adds_substituent():
    # Growing an ortho methyl is a swap for a larger, same-port fragment.
    tree = fragment(_embed("Cc1ccccc1"))
    swap(_ring_node(tree), "[*]c1ccccc1C")
    product = Chem.RemoveHs(tree.reconstruct())
    assert rdMolDescriptors.CalcMolFormula(product) == "C8H10"  # a xylene


def test_swap_preserves_untouched_ring_coordinates():
    mol = _embed("Cc1ccccc1")
    tree = fragment(mol)
    before = _aromatic_coords(mol)
    swap(_leaf(tree, 1), "[*]C(F)(F)F")
    after = _aromatic_coords(tree.reconstruct())
    assert before == after


def test_swap_without_conformer_is_connectivity_only():
    tree = fragment(prepare_molecule("Cc1ccccc1", three_d=False))  # no 3D coords
    swap(_leaf(tree, 1), "[*]Cl")
    assert _product_smiles(tree) == Chem.CanonSmiles("Clc1ccccc1")


def test_swap_is_undoable():
    tree = fragment(_embed("Cc1ccccc1"))
    leaf = _leaf(tree, 1)
    swap(leaf, "[*]C(F)(F)F")
    leaf.undo()
    assert _product_smiles(tree) == Chem.CanonSmiles("Cc1ccccc1")


def test_swap_rejects_port_count_mismatch():
    tree = fragment(_embed("Cc1ccc(C)cc1"))  # ring has two ports
    with pytest.raises(ValueError, match="port"):
        swap(_ring_node(tree), "[*]C(F)(F)F")
