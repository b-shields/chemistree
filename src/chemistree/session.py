"""Design session: direct, id-based editing of one molecule.

``DesignSession`` surfaces a precise map of every group and its atom positions
through :meth:`describe`, and edits by direct rdkit atom id — no fuzzy name or
position resolution. It fragments the molecule into a tree and normalizes every
fragment to explicit hydrogens so each atom (heavy or H) has a stable, addressable
id. Structural edits go through the tree's ``resplice`` so the tree always matches
what constructing from the edited molecule would give.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from chemistree.annotations import NodeAnnotation, annotate
from chemistree.chem import prepare_molecule
from chemistree.describe import describe_group, describe_tree
from chemistree.edits import grow_region, mutate_region, swap_region
from chemistree.fragment import Fragment
from chemistree.fragmenter import fragment
from chemistree.naming import group_smiles
from chemistree.receptor import Receptor, Residue
from chemistree.scoring import ScoreComponents, score_pose
from chemistree.torsion import ScanResult, axis_matrix, scan_torsion, worst_overlap
from chemistree.tree import Edge, FragmentNode, heavy_count


class DesignSession:
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
        """A compact markdown inventory of the current groups.

        Lists each group's id, name, fragment, role, and connections. Call
        :meth:`describe_group` for a group's atom positions, rings, and topology.
        When the session has a receptor and the ligand is posed in 3D, the current
        predicted affinity is appended, so it updates with every edit.
        """
        overview = describe_tree(self.tree)
        components = self._pose_components()
        if components is None:
            return overview
        return f"{overview}\n\n{_affinity_line(components)}"

    def describe_group(
        self, group_id: int, *, radius: int = 3, use_matrix: bool = False
    ) -> str:
        """The atom positions, rings, and neighbourhood of one group.

        Args:
            group_id: Id of the group to detail.
            radius: Farthest bond distance the positions section describes.
            use_matrix: Show the raw topology distance matrix instead of the
                chemist's-terms positions section (for comparison).

        Returns:
            A markdown section with the group's Atom Map, Rings, and positions.
        """
        return describe_group(self.tree, group_id, radius=radius, use_matrix=use_matrix)

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

    def distance(self, residue_id: str) -> str:
        """Report how close each group is to a receptor residue.

        Args:
            residue_id: A residue name ("PHE") or name with number ("PHE382").

        Returns:
            Markdown: a table of each group's minimum heavy-atom distance to the
            residue, sorted closest first, then a per-atom details section.

        Raises:
            ValueError: If no receptor is loaded or the ligand lacks 3D coordinates.
            NotFound: If no residue matches ``residue_id``.
        """
        if self.receptor is None:
            raise ValueError("spatial distances need a receptor")
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            raise ValueError("spatial distances need a ligand with 3D coordinates")
        residue = self.receptor.residue_atoms(residue_id)
        labels = {node.id: node for node in annotate(self.tree).nodes}
        groups = []
        for node in self.tree.nodes:
            assert node.id is not None
            annotation = labels[node.id]
            groups.append(
                _GroupDistances(
                    node_id=node.id,
                    label=annotation.name or annotation.classification,
                    per_atom=_atom_distances(node.current.mol, residue),
                )
            )
        groups.sort(key=lambda g: g.minimum)
        return _distance_report(residue_id, groups)

    def contacts(self, dist_cutoff: float = 4.5) -> str:
        """Report the closest group and atom for each binding-site residue.

        The binding site is every receptor residue with an atom within
        ``dist_cutoff`` of any ligand heavy atom. For each such residue this names
        the single closest ligand group and the atom in it (ports excluded), so a
        request that targets a residue maps straight to a group and atom to edit.

        Args:
            dist_cutoff: Site radius in angstrom: a residue counts when any of its
                atoms is within this distance of any ligand heavy atom.

        Returns:
            Markdown: a table of residue, closest group, closest atom, and their
            distance, sorted closest first.

        Raises:
            ValueError: If no receptor is loaded or the ligand lacks 3D coordinates.
        """
        if self.receptor is None:
            raise ValueError("spatial contacts need a receptor")
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            raise ValueError("spatial contacts need a ligand with 3D coordinates")
        labels = {node.id: node for node in annotate(self.tree).nodes}
        contacts = [
            self._closest_contact(
                residue, self.receptor.residue_positions(residue), labels
            )
            for residue in self.receptor.pocket(self.molecule(), within=dist_cutoff)
        ]
        contacts.sort(key=lambda c: c.distance)
        return _contacts_report(dist_cutoff, contacts)

    def _closest_contact(
        self,
        residue: Residue,
        coords: np.ndarray,
        labels: dict[int, NodeAnnotation],
    ) -> _Contact:
        """The closest group and atom of the ligand to one residue's atoms."""
        best: _Contact | None = None
        for node in self.tree.nodes:
            assert node.id is not None
            annotation = labels[node.id]
            for idx, symbol, distance in _atom_distances(node.current.mol, coords):
                if best is None or distance < best.distance:
                    best = _Contact(
                        residue=f"{residue.name}{residue.number}",
                        node_id=node.id,
                        label=annotation.name or annotation.classification,
                        atom_id=idx,
                        symbol=symbol,
                        distance=distance,
                    )
        assert best is not None  # a pocket residue has at least one nearby atom
        return best

    def rotate(self, group_id: int, degrees: float, *, window: float = 60.0) -> str:
        """Rotate a group about its attachment bond, settling to reduce clashes.

        The group turns about the single bond joining it to the rest of the
        molecule, carrying its own substituents as one rigid body. Every whole-
        degree turn is scored for steric clash (with the rest of the ligand and,
        when present, the receptor); the least-clashing turn within ``window`` of
        the request is applied. The constitution is unchanged, so the SMILES stays
        the same.

        Args:
            group_id: Id of the group to rotate.
            degrees: Requested turn, in degrees.
            window: Half-width, in degrees, of the search around ``degrees``.

        Returns:
            A short report: the applied turn, the clash score before and after,
            and the global-best turn when it would relieve the clash further.

        Raises:
            ValueError: If the ligand lacks 3D coordinates or the group is the
                only fragment.
        """
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            raise ValueError("rotation needs a ligand with 3D coordinates")
        node = self.tree.node(group_id)
        if not self.tree.neighbors(node):
            raise ValueError("cannot rotate the only group")
        axis_edge, _, moving = self._attachment(node)
        axis_point, axis_dir = self._axis(node, axis_edge)

        result = self._scan(self._pose(), moving, axis_point, axis_dir, degrees, window)
        applied = result.window_best[0]
        matrix = axis_matrix(axis_point, axis_dir, math.radians(applied))
        for moving_node in moving:
            rotated = Chem.Mol(moving_node.current.mol)
            rdMolTransforms.TransformConformer(rotated.GetConformer(), matrix)
            moving_node.push(Fragment(rotated))
        turned = list(moving)

        def undo() -> None:
            """Revert each turned fragment to its pre-rotation snapshot."""
            for moving_node in turned:
                moving_node.undo()

        self._undo_stack.append(undo)
        self._record()

        label = self._label(group_id)
        return _rotation_report(group_id, label, degrees, applied, result)

    def clashes(self, *, tol: float = 0.4) -> str:
        """Report steric clashes in the current pose, worst first.

        Lists every group pair whose atoms overlap past their van der Waals radii,
        and — when a receptor is loaded — every group that overlaps a residue. Atom
        pairs within two bonds of each other are normal geometry, not clashes, so
        they are excluded. This is the check to run after an edit, to decide
        whether to rotate a group to relieve a clash.

        Args:
            tol: Overlap allowed before a contact counts as a clash, in angstrom.

        Returns:
            Markdown: one line per clash, worst first, or a clear "no clashes" line.

        Raises:
            ValueError: If the ligand lacks 3D coordinates.
        """
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            raise ValueError("clash detection needs a ligand with 3D coordinates")
        labels = {n.id: n for n in annotate(self.tree).nodes}
        pose = self._pose()
        findings = self._ligand_clashes(pose, labels, tol) + self._receptor_clashes(
            pose, labels, tol
        )
        findings.sort(key=lambda f: -f[0])
        return _clashes_report(findings)

    def _pose_components(self) -> ScoreComponents | None:
        """The current pose's Vinardo score, or None without a posed receptor.

        Returns:
            The score breakdown when the session has a receptor and the ligand is
            posed in 3D; None otherwise, so callers can omit affinity cleanly.
        """
        if self.receptor is None:
            return None
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            return None
        coords, typing = self.receptor.scoring_context()
        return score_pose(self.molecule(), coords, typing)

    def _ligand_clashes(
        self, pose: _Pose, labels: dict[int, NodeAnnotation], tol: float
    ) -> list[tuple[float, str]]:
        """The worst overlap between each pair of groups, over two bonds apart."""
        overlap = pose.radii[:, None] + pose.radii[None, :] - tol - _distances(pose.xyz)
        far = pose.bond_dist > 2  # 1-2 and 1-3 contacts are normal geometry
        cross = pose.node_id[:, None] != pose.node_id[None, :]
        clashing = np.argwhere(np.triu(far & cross & (overlap > _CLASH_FLOOR), 1))

        worst: dict[frozenset[int], float] = {}
        for i, j in clashing:
            pair = frozenset((int(pose.node_id[i]), int(pose.node_id[j])))
            worst[pair] = max(worst.get(pair, 0.0), float(overlap[i, j]))
        findings = []
        for pair, value in worst.items():
            a, b = sorted(pair)
            findings.append(
                (
                    value,
                    f"[{a}] {_name(labels[a])} and [{b}] {_name(labels[b])} "
                    f"overlap by {value:.2f} A",
                )
            )
        return findings

    def _receptor_clashes(
        self, pose: _Pose, labels: dict[int, NodeAnnotation], tol: float
    ) -> list[tuple[float, str]]:
        """The worst residue overlap for each group, when a receptor is loaded."""
        if self.receptor is None:
            return []
        rec_xyz, rec_r = self.receptor.heavy_atoms()
        rec_labels = self.receptor.heavy_atom_labels()
        findings = []
        for group_id in np.unique(pose.node_id):
            mask = pose.node_id == group_id
            overlap, _, j = worst_overlap(
                pose.xyz[mask], pose.radii[mask], rec_xyz, rec_r, tol=tol
            )
            if overlap > _CLASH_FLOOR:
                findings.append(
                    (
                        overlap,
                        f"[{group_id}] {_name(labels[int(group_id)])} and "
                        f"{rec_labels[j]} overlap by {overlap:.2f} A",
                    )
                )
        return findings

    def _pose(self) -> _Pose:
        """The current whole-molecule heavy-atom pose, tagged by group.

        Returns:
            A pose with each heavy atom's coordinates, van der Waals radius, owning
            group id, and the bond-count distance matrix, all built from the
            reconstructed molecule so clash logic sees the real bond graph.
        """
        mol = self.molecule()
        table = Chem.GetPeriodicTable()
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        numbers = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy]
        node_id = [mol.GetAtomWithIdx(i).GetIntProp("node_id") for i in heavy]
        bond_dist = Chem.GetDistanceMatrix(mol)[np.ix_(heavy, heavy)]
        return _Pose(
            xyz=mol.GetConformer().GetPositions()[heavy],
            radii=np.array([table.GetRvdw(z) for z in numbers]),
            node_id=np.array(node_id),
            bond_dist=bond_dist,
        )

    def _attachment(
        self, node: FragmentNode
    ) -> tuple[Edge, FragmentNode, set[FragmentNode]]:
        """The bond to rotate about, its far node, and the moving-side nodes.

        The attachment bond is the incident edge whose near side (the component
        holding ``node``) has the fewest heavy atoms, so a leaf turns alone and a
        substituted ring turns with its substituents about the bond to the scaffold.
        """
        best: tuple[tuple[int, int], Edge, FragmentNode, set[FragmentNode]] | None = (
            None
        )
        for edge, other in self.tree.neighbors(node):
            side = self._moving_side(node, edge)
            key = (sum(heavy_count(n.current.mol) for n in side), edge.label)
            if best is None or key < best[0]:
                best = (key, edge, other, side)
        assert best is not None  # a non-only node has at least one incident edge
        return best[1], best[2], best[3]

    def _moving_side(self, node: FragmentNode, axis_edge: Edge) -> set[FragmentNode]:
        """Nodes reachable from ``node`` without crossing ``axis_edge``."""
        seen = {node}
        stack = [node]
        while stack:
            current = stack.pop()
            for edge, other in self.tree.neighbors(current):
                if edge is axis_edge or other in seen:
                    continue
                seen.add(other)
                stack.append(other)
        return seen

    def _axis(
        self, node: FragmentNode, axis_edge: Edge
    ) -> tuple[np.ndarray, np.ndarray]:
        """The rotation axis: a point on it and its direction, from ``node``'s port."""
        port = next(p for p in node.current.ports if p.label == axis_edge.label)
        conf = node.current.mol.GetConformer()
        point = np.array(list(conf.GetAtomPosition(port.anchor_idx)))
        direction = np.array(list(conf.GetAtomPosition(port.dummy_idx))) - point
        return point, direction

    def _scan(
        self,
        pose: _Pose,
        moving: set[FragmentNode],
        axis_point: np.ndarray,
        axis_dir: np.ndarray,
        degrees: float,
        window: float,
    ) -> ScanResult:
        """Score every one-degree turn of the moving side against its surroundings.

        The moving atoms are the group's subtree; the surroundings are the rest of
        the ligand more than two bonds away (nearer atoms hold fixed geometry the
        turn cannot change) plus any nearby receptor atoms.
        """
        moving_ids = {n.id for n in moving}
        is_moving = np.isin(pose.node_id, list(moving_ids))
        # Keep a static atom only if it is over two bonds from every moving atom.
        near_moving = (pose.bond_dist[~is_moving][:, is_moving] <= 2).any(axis=1)
        other_xyz = pose.xyz[~is_moving][~near_moving]
        other_r = pose.radii[~is_moving][~near_moving]

        moving_xyz = pose.xyz[is_moving]
        other_xyz, other_r = self._add_receptor(
            other_xyz, other_r, moving_xyz, axis_point
        )
        return scan_torsion(
            moving_xyz,
            axis_point=axis_point,
            axis_dir=axis_dir,
            moving_r=pose.radii[is_moving],
            other_xyz=other_xyz,
            other_r=other_r,
            target_deg=degrees,
            window_deg=window,
        )

    def _add_receptor(
        self,
        other_xyz: np.ndarray,
        other_r: np.ndarray,
        moving_xyz: np.ndarray,
        axis_point: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append receptor heavy atoms that any turn could bring near the group."""
        if self.receptor is None or len(moving_xyz) == 0:
            return other_xyz, other_r
        rec_xyz, rec_r = self.receptor.heavy_atoms()
        reach = float(np.linalg.norm(moving_xyz - axis_point, axis=1).max())
        near = np.linalg.norm(rec_xyz - axis_point, axis=1) < reach + 4.0
        if not near.any():
            return other_xyz, other_r
        return (
            np.vstack([other_xyz, rec_xyz[near]]),
            np.concatenate([other_r, rec_r[near]]),
        )

    def _label(self, group_id: int) -> str:
        """The chemist-facing name of a group, for reports."""
        return _name({n.id: n for n in annotate(self.tree).nodes}[group_id])

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


@dataclass(frozen=True)
class _GroupDistances:
    """One group's distances to a residue: a label and per-heavy-atom distances."""

    node_id: int | None
    label: str
    per_atom: list[tuple[int, str, float]]

    @property
    def minimum(self) -> float:
        """The closest heavy-atom approach of this group to the residue."""
        return min(distance for _, _, distance in self.per_atom)


# Overlaps below this (angstrom, past the tolerance) are incidental contact, not
# a clash worth reporting.
_CLASH_FLOOR = 0.1


@dataclass(frozen=True)
class _Pose:
    """The current molecule's heavy atoms, ready for clash geometry.

    Attributes:
        xyz: (N, 3) heavy-atom coordinates.
        radii: (N,) van der Waals radii.
        node_id: (N,) owning group id per atom.
        bond_dist: (N, N) shortest bond-count distance between heavy atoms, so
            pairs within two bonds (normal geometry) can be excluded.
    """

    xyz: np.ndarray
    radii: np.ndarray
    node_id: np.ndarray
    bond_dist: np.ndarray


def _distances(xyz: np.ndarray) -> np.ndarray:
    """The (N, N) matrix of pairwise Euclidean distances."""
    diff = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def _name(annotation: NodeAnnotation) -> str:
    """The chemist-facing name of a group: its curated name, else its class."""
    return annotation.name or annotation.classification


def _affinity_line(components: ScoreComponents) -> str:
    """The predicted-affinity line appended to a receptor-posed describe."""
    return (
        f"**Predicted affinity (Vinardo):** {components.total:.2f} " "(lower is better)"
    )


def _clashes_report(findings: list[tuple[float, str]]) -> str:
    """Render the clash findings, already sorted worst first."""
    if not findings:
        return "# Clashes\n\nNo clashes."
    return "\n".join(["# Clashes", ""] + [f"- {desc}" for _, desc in findings])


def _rotation_report(
    group_id: int, label: str, requested: float, applied: int, result: ScanResult
) -> str:
    """Render the outcome of a torsion rotation for the agent."""
    before, after = result.current, result.window_best[1]
    lines = [
        f"Rotated [{group_id}] {label} by {applied} deg (requested {requested:g} deg).",
        f"Clash score {before:.2f} -> {after:.2f}.",
    ]
    global_deg, global_score = result.global_best
    if global_score < after - 0.05:
        lines.append(
            f"A larger turn to {global_deg} deg would lower the clash to "
            f"{global_score:.2f}."
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class _Contact:
    """One binding-site residue's closest ligand group and atom."""

    residue: str
    node_id: int
    label: str
    atom_id: int
    symbol: str
    distance: float


def _contacts_report(dist_cutoff: float, contacts: list[_Contact]) -> str:
    """Render the binding-site contacts table, contacts already sorted."""
    lines = [
        f"# Binding-site contacts (within {dist_cutoff:g} A)",
        "",
    ]
    if not contacts:
        lines.append(f"No residue within {dist_cutoff:g} A of the molecule.")
        return "\n".join(lines)
    lines += ["| residue | group | closest atom | distance (A) |", "|---|---|---|---|"]
    for contact in contacts:
        lines.append(
            f"| {contact.residue} | [{contact.node_id}] {contact.label} "
            f"| atom {contact.atom_id} ({contact.symbol}) | {contact.distance:.2f} |"
        )
    return "\n".join(lines)


def _atom_distances(mol: Chem.Mol, residue: np.ndarray) -> list[tuple[int, str, float]]:
    """Each heavy atom's id, symbol, and minimum distance to the residue atoms."""
    conf = mol.GetConformer()
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = np.array(list(conf.GetAtomPosition(atom.GetIdx())))
        closest = float(np.sqrt(((residue - position) ** 2).sum(-1)).min())
        out.append((atom.GetIdx(), atom.GetSymbol(), closest))
    return out


def _distance_report(residue_id: str, groups: list[_GroupDistances]) -> str:
    """Render the distance table and per-group details, groups already sorted."""
    lines = [
        f"# Distances to {residue_id}",
        "",
        "| group | min distance (A) |",
        "|---|---|",
    ]
    for group in groups:
        lines.append(f"| [{group.node_id}] {group.label} | {group.minimum:.2f} |")
    lines += ["", "## Details"]
    for group in groups:
        lines.append(f"### [{group.node_id}] {group.label}")
        for idx, symbol, distance in group.per_atom:
            lines.append(f"- atom {idx} ({symbol}): {distance:.2f}")
    return "\n".join(lines)


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
