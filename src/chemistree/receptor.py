"""Receptor context: resolve ligand fragments by proximity to protein residues.

The receptor is a read-only RDKit molecule (e.g. from a PDB) with per-atom residue
info and a 3D conformer. When the ligand is posed in the same frame, an ambiguous
reference like "the methyl near the PHE" resolves deterministically by distance.
This module owns all of that geometry; higher layers only call it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from chemistree.errors import NotFound
from chemistree.tree import FragmentNode

# Default proximity radius (angstrom). Loose enough to grow toward a nearby residue,
# tight enough to reject residues in another part of the protein.
_WITHIN = 8.0


@dataclass(frozen=True)
class Residue:
    """A protein residue and the atoms that belong to it.

    Attributes:
        name: Residue name, e.g. "PHE".
        number: Residue sequence number.
        chain: Chain identifier.
        atoms: Indices of the residue's atoms in the receptor molecule.
    """

    name: str
    number: int
    chain: str
    atoms: tuple[int, ...]


def _parse_residue_spec(spec: str) -> tuple[str, int | None]:
    """Split a residue spec into its name and optional trailing number.

    "ALA37" -> ("ALA", 37); "ALA" -> ("ALA", None). The name is upper-cased so
    "ala37" resolves too.

    Args:
        spec: A residue name, optionally with a sequence number appended.

    Returns:
        The residue name and its number, or None when no number is given.
    """
    text = spec.strip()
    split = len(text)
    while split > 0 and text[split - 1].isdigit():
        split -= 1
    name = text[:split].upper()
    number = int(text[split:]) if split < len(text) else None
    return name, number


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two sets of points.

    Args:
        a: An (N, 3) array of coordinates.
        b: An (M, 3) array of coordinates.

    Returns:
        The minimum pairwise Euclidean distance.
    """
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((diff * diff).sum(-1)).min())


class Receptor:
    """A protein structure used as read-only context for proximity resolution."""

    def __init__(self, mol: Chem.Mol):
        """Wrap a receptor molecule and group its atoms into residues once.

        Args:
            mol: The protein molecule, with per-atom PDB residue info and a
                3D conformer, typically read from a PDB.
        """
        self.mol = mol
        self._residues = _group_residues(mol)

    def residues(self, name: str | None = None) -> list[Residue]:
        """The receptor's residues, optionally filtered by name (and number).

        The filter accepts a bare name ("ALA", every alanine) or a name with a
        sequence number ("ALA37", the one residue) — the form a chemist writes.

        Args:
            name: If given, a residue name, optionally with its number appended.

        Returns:
            The matching residues.
        """
        if name is None:
            return list(self._residues)
        want_name, want_number = _parse_residue_spec(name)
        return [
            residue
            for residue in self._residues
            if residue.name == want_name
            and (want_number is None or residue.number == want_number)
        ]

    def nearest(
        self, candidates: list[FragmentNode], residue: str, *, within: float = _WITHIN
    ) -> FragmentNode:
        """The candidate fragment closest to a residue of the given name.

        Mutual nearest neighbor: over every (fragment, residue) pair, the smallest
        interatomic distance selects the fragment. Candidates and residues are
        visited in a stable order, so ties resolve reproducibly.

        Args:
            candidates: Ligand nodes to choose among (each posed in 3D).
            residue: Residue name to measure distance to (e.g. "PHE").
            within: Reject the match if the closest pair is farther than this many
                angstrom, so a residue in another region cannot resolve a reference.

        Returns:
            The closest candidate node.

        Raises:
            NotFound: If no candidate or residue is present, or none is within range.
        """
        if not candidates:
            raise NotFound("no candidate fragments")
        targets = self.residues(residue)
        if not targets:
            raise NotFound(f"no {residue} residue in the receptor")

        target_coords = [
            self._positions(res.atoms)
            for res in sorted(targets, key=lambda r: r.number)
        ]
        best: FragmentNode | None = None
        best_distance = float("inf")
        for node in sorted(candidates, key=_node_key):
            fragment_coords = _heavy_positions(node.current.mol)
            for coords in target_coords:
                distance = min_distance(fragment_coords, coords)
                if distance < best_distance:
                    best_distance = distance
                    best = node
        if best is None or best_distance > within:
            raise NotFound(f"no {residue} within {within:g} A of the candidates")
        return best

    def pocket(self, ligand: Chem.Mol, within: float = 8.0) -> list[Residue]:
        """Residues with any atom within a distance of the ligand.

        Args:
            ligand: The ligand molecule, posed in the receptor frame.
            within: Distance cutoff in angstrom.

        Returns:
            The residues lining the binding site, in residue order.
        """
        ligand_coords = _heavy_positions(ligand)
        return [
            residue
            for residue in self._residues
            if min_distance(self._positions(residue.atoms), ligand_coords) < within
        ]

    def _positions(self, atoms: tuple[int, ...]) -> np.ndarray:
        """Coordinates of the given receptor atoms as an (N, 3) array."""
        conf = self.mol.GetConformer()
        return np.array([list(conf.GetAtomPosition(i)) for i in atoms])


def _group_residues(mol: Chem.Mol) -> list[Residue]:
    """Group a molecule's atoms into residues by their PDB residue info."""
    grouped: dict[tuple[str, int, str], list[int]] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (
            info.GetChainId(),
            info.GetResidueNumber(),
            info.GetResidueName().strip(),
        )
        grouped.setdefault(key, []).append(atom.GetIdx())
    return [
        Residue(name, number, chain, tuple(atoms))
        for (chain, number, name), atoms in grouped.items()
    ]


def _node_key(node: FragmentNode) -> int:
    return -1 if node.id is None else node.id


def _heavy_positions(mol: Chem.Mol) -> np.ndarray:
    """Coordinates of a fragment's heavy atoms as an (N, 3) array."""
    conf = mol.GetConformer()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return np.array([list(conf.GetAtomPosition(i)) for i in heavy])
