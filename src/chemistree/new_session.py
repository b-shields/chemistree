"""Redesigned design session: direct, id-based editing of one molecule.

``NewDesignSession`` surfaces a precise map of every group and its atom positions
through :meth:`describe`, and edits by direct rdkit atom id — no fuzzy name or
position resolution. It fragments the molecule into a tree and normalizes every
fragment to explicit hydrogens so each atom (heavy or H) has a stable, addressable
id. Structural edits go through the tree's ``resplice`` so the tree always matches
what constructing from the edited molecule would give.
"""

from __future__ import annotations

from collections.abc import Callable

from rdkit import Chem

from chemistree.chem import prepare_molecule
from chemistree.describe import describe_tree
from chemistree.edits import grow_region, mutate_region, swap_region
from chemistree.fragment import Fragment
from chemistree.fragmenter import fragment
from chemistree.naming import group_smiles
from chemistree.receptor import Receptor
from chemistree.tree import FragmentNode


class NewDesignSession:
    """A molecule being edited, with stable node and atom ids."""

    def __init__(
        self,
        molecule: str | Chem.Mol,
        receptor: Chem.Mol | None = None,
        *,
        three_d: bool = True,
    ):
        """Prepare a molecule, fragment it, and normalize fragments to explicit H.

        Args:
            molecule: A SMILES string or Mol for the ligand.
            receptor: Optional protein (e.g. from a PDB) as read-only context.
            three_d: Prepare the ligand with explicit hydrogens and a 3D conformer.
        """
        self.tree = fragment(prepare_molecule(molecule, three_d=three_d))
        self._normalize_hydrogens()
        self.receptor = Receptor(receptor) if receptor is not None else None
        self._undo_stack: list[Callable[[], None]] = []
        self.smiles_history: list[str] = [self.smiles()]

    def _normalize_hydrogens(self) -> None:
        """Make every fragment's hydrogens explicit, so each atom has a stable id.

        Coordinates are added only when the fragment already has a conformer, so
        the 2D-only path still gets addressable hydrogens without a bond vector.
        """
        for node in self.tree.nodes:
            mol = node.current.mol
            explicit = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)
            node.history[0] = Fragment(explicit)

    def _record(self) -> None:
        """Append the current SMILES to the history, in edit order."""
        self.smiles_history.append(self.smiles())

    def describe(self) -> str:
        """A readable markdown map of the current fragments and atom positions."""
        return describe_tree(self.tree)

    def molecule(self) -> Chem.Mol:
        """The current molecule, reconstructed from the tree."""
        return self.tree.reconstruct()

    def smiles(self) -> str:
        """Canonical SMILES of the current molecule, without explicit hydrogens."""
        return str(Chem.MolToSmiles(Chem.RemoveHs(self.tree.reconstruct())))

    def swap(self, group_id: int, group: str | Chem.Mol) -> None:
        """Replace a group's fragment with a new group.

        Args:
            group_id: Id of the group to replace.
            group: A curated group name or a SMILES/Mol with matching ports.
        """
        node = self.tree.node(group_id)
        region = swap_region(node.current, _as_group(group))
        self._apply(node, region)

    def grow(self, group_id: int, position_id: int, group: str | Chem.Mol) -> int:
        """Grow a group where a hydrogen is, following fragmentation rules.

        The group takes the place of the hydrogen at ``position_id`` (its bond
        vector is the new group's exit vector). Whether the group becomes its own
        node or merges into the target follows the same policy as construction.

        Args:
            group_id: Id of the group bearing the hydrogen.
            position_id: Atom id of the hydrogen to replace.
            group: A curated group name or a SMILES/Mol with one port.

        Returns:
            Id of the group that now carries the grown group (a new node when the
            growth splits off, else ``group_id``).

        Raises:
            ValueError: If ``position_id`` is not a hydrogen of the group.
        """
        node = self.tree.node(group_id)
        region = grow_region(node.current, position_id, _as_group(group))
        sub_nodes = self._apply(node, region)
        grown = max(sub_nodes, key=lambda n: n.id if n.id is not None else -1)
        assert grown.id is not None
        return grown.id

    def mutate(self, group_id: int, position_id: int, element: str) -> int:
        """Change one heavy atom's element, addressing it by its atom id.

        Args:
            group_id: Id of the group to edit.
            position_id: Atom id of the heavy atom to change.
            element: New element, as a symbol ("N") or name ("nitrogen").

        Returns:
            The mutated atom id (``position_id``).

        Raises:
            ValueError: If ``position_id`` is not a heavy atom, or the change
                leaves an invalid valence.
        """
        node = self.tree.node(group_id)
        atom = node.current.mol.GetAtomWithIdx(position_id)
        if atom.GetAtomicNum() <= 1:
            raise ValueError(
                f"atom {position_id} is not a heavy atom; mutate changes a heavy atom"
            )
        region = mutate_region(node.current, position_id, _element_number(element))
        self._apply(node, region)
        return position_id

    def remove(self, group_id: int) -> None:
        """Delete a group and its subtree, capping the parent with hydrogen.

        Args:
            group_id: Id of the group to remove.

        Raises:
            ValueError: If the group is the root scaffold.
        """
        removed = self.tree.remove_subtree(self.tree.node(group_id))
        self._undo_stack.append(lambda: self.tree.restore_subtree(removed))
        self._record()

    def _apply(self, node: FragmentNode, region_mol: Chem.Mol) -> list[FragmentNode]:
        """Resplice an edited region into the tree, recording undo and history."""
        sub_nodes, undo = self.tree.resplice(node, region_mol)
        self._normalize_new_hydrogens(sub_nodes)
        self._undo_stack.append(undo)
        self._record()
        return sub_nodes

    def _normalize_new_hydrogens(self, nodes: list[FragmentNode]) -> None:
        """Make hydrogens explicit on freshly spliced fragments, for stable ids."""
        for node in nodes:
            mol = node.current.mol
            explicit = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)
            node.history[-1] = Fragment(explicit)

    def undo(self) -> None:
        """Revert the most recent edit.

        Raises:
            ValueError: If there is nothing to undo.
        """
        if not self._undo_stack:
            raise ValueError("nothing to undo")
        self._undo_stack.pop()()
        self._record()


_ELEMENT_NAMES = {
    "carbon": 6,
    "nitrogen": 7,
    "oxygen": 8,
    "fluorine": 9,
    "phosphorus": 15,
    "sulfur": 16,
}


def _element_number(element: str) -> int:
    """Atomic number for an element symbol ("N") or common name ("nitrogen").

    Raises:
        ValueError: If the element is not recognized.
    """
    key = element.strip().lower()
    if key in _ELEMENT_NAMES:
        return _ELEMENT_NAMES[key]
    try:
        number = Chem.GetPeriodicTable().GetAtomicNumber(element.strip().capitalize())
    except RuntimeError as error:
        raise ValueError(f"unknown element: {element!r}") from error
    if number <= 0:
        raise ValueError(f"unknown element: {element!r}")
    return int(number)


def _as_group(group: str | Chem.Mol) -> str | Chem.Mol:
    """Resolve a curated group name to SMILES; pass SMILES/Mol through unchanged.

    Raises:
        ValueError: If a string is neither a known group name nor valid SMILES.
    """
    if not isinstance(group, str):
        return group
    smiles = group_smiles(group)
    if smiles is not None:
        return smiles
    if Chem.MolFromSmiles(group) is None:
        raise ValueError(
            f"unknown group {group!r}: not a known group name and not valid SMILES"
        )
    return group
