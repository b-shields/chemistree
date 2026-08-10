"""High-level design session: one stateful object per molecule.

``DesignSession`` owns a fragment tree (and an optional receptor) so it can be held
in a single global and persist across LLM tool calls. It is a thin boundary: the
caller passes names, ids, and positions; the session resolves them deterministically
and runs the edit. Group arguments accept a curated name ("isopropyl") or a SMILES.
"""

from __future__ import annotations

from rdkit import Chem

from chemistree.annotations import annotate
from chemistree.chem import prepare_molecule
from chemistree.edits import add_substituent
from chemistree.edits import swap as _swap_fragment
from chemistree.errors import NotFound
from chemistree.fragmenter import fragment
from chemistree.naming import group_smiles
from chemistree.receptor import Receptor
from chemistree.selection import resolve_site, select, select_one


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
        _swap_fragment(self.tree.node(node_id), _as_group(group))

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
        substituent = select_one(
            self.tree, description=reference, name=reference, neighbor_of=scaffold
        )
        site = resolve_site(self.tree, scaffold, substituent, position)
        add_substituent(scaffold, site, _as_group(group))
        return site

    def undo(self, node_id: int) -> None:
        """Revert a node's most recent edit.

        Args:
            node_id: Id of the node to revert.
        """
        self.tree.node(node_id).undo()

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


def _as_group(group: str | Chem.Mol) -> str | Chem.Mol:
    """Resolve a curated group name to SMILES; pass SMILES/Mol through unchanged."""
    if isinstance(group, str):
        return group_smiles(group) or group
    return group
