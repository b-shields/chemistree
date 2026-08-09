"""Stereochemistry: swaps conserve chiral volume unless inversion is stated."""

from rdkit import Chem
from rdkit.Chem import AllChem

from chemistree import fragment, swap
from chemistree.geometry import chiral_volume


def _embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


def _sign(x: float) -> int:
    return 1 if x > 0 else -1


def _center_sign(mol: Chem.Mol, smarts: str) -> int:
    """Sign of the chiral volume at a matched center, over two named neighbors + H.

    The SMARTS matches (center, neighbor, neighbor); the third reference is the
    center's hydrogen. Using neighbors unchanged by the edit makes the sign
    comparable across the original and product molecules.
    """
    center, n1, n2 = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))[:3]
    h = next(
        a.GetIdx()
        for a in mol.GetAtomWithIdx(center).GetNeighbors()
        if a.GetAtomicNum() == 1
    )
    return _sign(chiral_volume(mol.GetConformer(), center, n1, n2, h))


def _nonaromatic_node(tree):
    return next(
        node
        for node in tree.nodes
        if not any(a.GetIsAromatic() for a in node.current.mol.GetAtoms())
    )


BENZYLIC = "[CX4]([CH3])[c]"  # 1-phenylethyl center, methyl, aromatic C
CARBINOL = "[CX4]([CH3])[OX2]"  # 2-hydroxy center, methyl, oxygen


def test_stereo_conserved_at_connection_point():
    # The stereocenter is the port anchor; swapping Cl->F must keep its handedness.
    before = _center_sign(_embed("C[C@H](Cl)c1ccccc1"), BENZYLIC)
    tree = fragment(_embed("C[C@H](Cl)c1ccccc1"))
    swap(_nonaromatic_node(tree), "[*]C(C)F")
    assert _center_sign(tree.reconstruct(), BENZYLIC) == before


def test_stereo_conserved_at_distant_center():
    # A remote center inside the fragment stays put while an atom near the port changes.
    before = _center_sign(_embed("C[C@H](O)CCc1ccccc1"), CARBINOL)
    tree = fragment(_embed("C[C@H](O)CCc1ccccc1"))
    swap(_nonaromatic_node(tree), "[*]C(F)CC(C)O")
    assert _center_sign(tree.reconstruct(), CARBINOL) == before


def test_explicit_stereo_overrides_conservation():
    # Stating the opposite configuration inverts that center; the other tag conserves.
    original = _center_sign(_embed("C[C@H](O)CCc1ccccc1"), CARBINOL)

    def product_sign(tag: str) -> int:
        tree = fragment(_embed("C[C@H](O)CCc1ccccc1"))
        swap(_nonaromatic_node(tree), f"[*]C(F)C[{tag}](C)O")
        return _center_sign(tree.reconstruct(), CARBINOL)

    cw, ccw = product_sign("C@H"), product_sign("C@@H")
    assert cw == -ccw  # explicit tag controls handedness
    assert {cw, ccw} == {original, -original}  # one conserves, one inverts
