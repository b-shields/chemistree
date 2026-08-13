"""High-level design session: one stateful object per molecule.

``DesignSession`` owns a fragment tree (and an optional receptor) so it can be held
in a single global and persist across LLM tool calls. It is a thin boundary: the
caller passes names, ids, and positions; the session resolves them deterministically
and runs the edit. Group arguments accept a curated name ("isopropyl") or a SMILES.
"""

from __future__ import annotations

from collections.abc import Callable

from rdkit import Chem

from chemistree.annotations import annotate
from chemistree.chem import prepare_molecule
from chemistree.edits import add_substituent, mutate_atom
from chemistree.edits import swap as _swap_fragment
from chemistree.errors import NotFound
from chemistree.fragmenter import fragment
from chemistree.naming import group_smiles
from chemistree.receptor import Receptor
from chemistree.selection import (
    resolve_between,
    resolve_shared_site,
    resolve_site,
    select,
    select_one,
)


class DesignSession:
    """A molecule being edited, with stable node ids and optional receptor context."""

    def __init__(
        self,
        molecule: str | Chem.Mol,
        receptor: Chem.Mol | None = None,
        *,
        three_d: bool = True,
    ):
        """Prepare a molecule and fragment it into an editable tree.

        Args:
            molecule: A SMILES string or Mol for the ligand.
            receptor: Optional protein (e.g. from a PDB) as read-only context.
            three_d: Prepare the ligand with explicit hydrogens and a 3D conformer.
        """
        self.tree = fragment(prepare_molecule(molecule, three_d=three_d))
        self.receptor = Receptor(receptor) if receptor is not None else None
        self._undo_stack: list[Callable[[], None]] = []

    def describe(self) -> str:
        """A readable markdown summary of the current fragments."""
        return annotate(self.tree).to_markdown()

    def annotations(self, *, atoms: bool = False) -> dict:
        """A JSON-serializable description of the current fragments.

        Args:
            atoms: Include per-atom detail on each node.

        Returns:
            The annotation as nested dicts.
        """
        return annotate(self.tree, atoms=atoms).to_dict()

    def molecule(self) -> Chem.Mol:
        """The current molecule, reconstructed from the tree."""
        return self.tree.reconstruct()

    def smiles(self) -> str:
        """Canonical SMILES of the current molecule, without explicit hydrogens."""
        return str(Chem.MolToSmiles(Chem.RemoveHs(self.tree.reconstruct())))

    def find(
        self, *, name: str | None = None, classification: str | None = None
    ) -> list[int]:
        """Ids of nodes matching a name and/or coarse class.

        Args:
            name: Required common name (see ``name_fragment``).
            classification: Required coarse class (see ``classify_fragment``).

        Returns:
            The matching node ids, in tree order.
        """
        matches = select(self.tree, name=name, classification=classification)
        return [node.id for node in matches if node.id is not None]

    def swap(self, node_id: int, group: str | Chem.Mol) -> None:
        """Replace a node's fragment with a new group.

        Args:
            node_id: Id of the node to edit.
            group: A curated group name or a SMILES/Mol with matching ports.
        """
        node = self.tree.node(node_id)
        _swap_fragment(node, _as_group(group))
        self._undo_stack.append(node.undo)

    def add(
        self,
        scaffold_id: int,
        group: str | Chem.Mol,
        *,
        position: int | str,
        reference: str,
    ) -> int:
        """Grow a group at a position relative to one of the scaffold's substituents.

        Args:
            scaffold_id: Id of the scaffold node to grow from.
            group: A curated group name or a SMILES/Mol with one port.
            position: A bond count, or a ring synonym (ortho/meta/para).
            reference: Name of the scaffold substituent to count from.

        Returns:
            Index of the scaffold atom the group was grown at.

        Raises:
            NotFound: If the reference or a valid site is not found.
            Ambiguous: If the reference or the site is not unique.
        """
        scaffold = self.tree.node(scaffold_id)
        substituents = select(self.tree, name=reference, neighbor_of=scaffold)
        if not substituents:
            raise NotFound(f"no {reference} on the scaffold")
        site = resolve_shared_site(self.tree, scaffold, substituents, position)
        add_substituent(scaffold, site, _as_group(group))
        self._undo_stack.append(scaffold.undo)
        return site

    def mutate(
        self,
        scaffold_id: int,
        element: str,
        *,
        between: tuple[str, str] | None = None,
        reference: str | None = None,
        position: int | str | None = None,
    ) -> int:
        """Change one ring atom's element, addressing it by its neighbors.

        The target atom is either the one between two named substituents
        (``between``), or one at a ``position`` offset from a single
        ``reference``. This is the atom-level path to ring heteroatom edits, e.g.
        turning the ring CH between an amino and a methyl into N to make a
        2-aminopyridine.

        Args:
            scaffold_id: Id of the ring node to edit.
            element: New element, as a symbol ("N") or name ("nitrogen").
            between: Names of two substituents the target atom sits between.
            reference: Name of one substituent to count from (with ``position``).
            position: A bond count, or a ring synonym, from ``reference``.

        Returns:
            Index of the mutated atom in the scaffold's fragment.

        Raises:
            ValueError: If the addressing is incomplete or the element is unknown.
            NotFound: If no such atom exists. Ambiguous: If it is not unique.
        """
        scaffold = self.tree.node(scaffold_id)
        if between is not None:
            first = select_one(
                self.tree, description=between[0], name=between[0], neighbor_of=scaffold
            )
            second = select_one(
                self.tree, description=between[1], name=between[1], neighbor_of=scaffold
            )
            atom = resolve_between(self.tree, scaffold, first, second)
        elif reference is not None and position is not None:
            substituent = select_one(
                self.tree, description=reference, name=reference, neighbor_of=scaffold
            )
            atom = resolve_site(self.tree, scaffold, substituent, position)
        else:
            raise ValueError("mutate needs between=(a, b) or reference and position")
        mutate_atom(scaffold, atom, _element_number(element))
        self._undo_stack.append(scaffold.undo)
        return atom

    def remove(self, node_id: int) -> None:
        """Delete a node and its subtree, capping the parent with hydrogen.

        "Delete the phenol ring" removes the ring and its own substituents and
        caps the anilino N to -NH2. The core scaffold (the tree's root) cannot be
        removed this way.

        Args:
            node_id: Id of the node to remove, along with everything hanging off
                it away from the core.

        Raises:
            ValueError: If the node is the root scaffold.
        """
        removed = self.tree.remove_subtree(self.tree.node(node_id))
        self._undo_stack.append(lambda: self.tree.restore_subtree(removed))

    def undo(self) -> None:
        """Revert the most recent edit (swap, add, mutate, or remove).

        Raises:
            ValueError: If there is nothing to undo.
        """
        if not self._undo_stack:
            raise ValueError("nothing to undo")
        self._undo_stack.pop()()

    def nearest(self, name: str, residue: str) -> int:
        """Id of the named fragment closest to a named receptor residue.

        The distance resolution lives in ``Receptor``; the session only checks its
        preconditions and selects the candidates.

        Args:
            name: Common name of the ligand fragment (e.g. "methyl").
            residue: Residue name in the receptor (e.g. "PHE").

        Returns:
            Id of the closest matching node.

        Raises:
            ValueError: If no receptor is loaded or the ligand lacks 3D coordinates.
            NotFound: If no matching fragment or residue exists.
        """
        if self.receptor is None:
            raise ValueError("spatial resolution needs a receptor")
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            raise ValueError("spatial resolution needs a ligand with 3D coordinates")
        candidates = select(self.tree, name=name)
        if not candidates:
            raise NotFound(f"no {name} in the ligand")
        node = self.receptor.nearest(candidates, residue)
        assert node.id is not None
        return node.id


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
            The message names the token so the agent can retry a synonym or a
            SMILES instead of seeing a confusing downstream parse error.
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
