"""Growing a substituent at a resolved position (the resolver end to end)."""

from rdkit import Chem

from chemistree import add_substituent, fragment, prepare_molecule
from chemistree.selection import (
    is_free_aromatic_carbon,
    resolve_site,
    select_one,
)


def _product(tree) -> str:
    return str(Chem.MolToSmiles(Chem.RemoveHs(tree.reconstruct())))


def test_grow_isopropyl_para_to_the_methyl():
    # "add an isopropyl para to the methyl on the benzene" -> p-cymene.
    tree = fragment(prepare_molecule("Cc1ccccc1"))
    ring = select_one(tree, description="phenyl", name="phenyl")
    methyl = select_one(tree, description="methyl", name="methyl", neighbor_of=ring)

    site = resolve_site(tree, ring, methyl, "para", site=is_free_aromatic_carbon)
    add_substituent(ring, site, "[*]C(C)C")

    assert _product(tree) == Chem.CanonSmiles("CC(C)c1ccc(C)cc1")


def test_grow_is_undoable():
    tree = fragment(prepare_molecule("Cc1ccccc1"))
    ring = select_one(tree, description="phenyl", name="phenyl")
    methyl = select_one(tree, description="methyl", name="methyl", neighbor_of=ring)

    site = resolve_site(tree, ring, methyl, "para", site=is_free_aromatic_carbon)
    add_substituent(ring, site, "[*]F")
    ring.undo()

    assert _product(tree) == Chem.CanonSmiles("Cc1ccccc1")
