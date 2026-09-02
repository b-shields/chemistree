"""Design session: direct, id-based editing of one molecule.

``DesignSession`` surfaces a precise map of every group and its atom positions
through :meth:`describe`, and edits by direct rdkit atom id — no fuzzy name or
position resolution. It fragments the molecule into a tree and normalizes every
fragment to explicit hydrogens so each atom (heavy or H) has a stable, addressable
id. Structural edits go through the tree's ``resplice`` so the tree always matches
what constructing from the edited molecule would give.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolTransforms

from chemistree.annotations import NodeAnnotation, annotate
from chemistree.chem import prepare_molecule
from chemistree.describe import describe_group, describe_tree
from chemistree.edits import grow_region, mutate_region, swap_region
from chemistree.fragment import Fragment
from chemistree.fragmenter import fragment
from chemistree.geometry import align_transform
from chemistree.heterocycles import (
    HETEROCYCLES,
    heterocycle_name,
    heterocycle_smiles,
    number_ring_system,
    ring_ports_summary,
)
from chemistree.naming import group_smiles
from chemistree.properties import profile_markdown
from chemistree.receptor import Receptor, Residue, min_distance
from chemistree.scoring import (
    ScoreComponents,
    atom_typing,
    intramolecular_mask,
    score_intramolecular,
    score_pose,
    vinardo_objective,
)
from chemistree.torsion import ScanResult, axis_matrix, scan_result, worst_overlap
from chemistree.tree import Edge, FragmentNode, FragmentTree, heavy_count


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
        self._snapshots: list[FragmentTree] = []
        self.smiles_history: list[str] = [self.smiles()]
        self._history: list[Chem.Mol] = [self.molecule()]

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
        """Record the current molecule, unless its SMILES was already seen.

        A molecule reached again (e.g. by growing a group then removing it) is
        not recorded twice, so the history holds only distinct molecules in the
        order first seen. ``smiles_history`` and :meth:`history` stay parallel.
        """
        smiles = self.smiles()
        if smiles in self.smiles_history:
            return
        self.smiles_history.append(smiles)
        self._history.append(self.molecule())

    def history(self) -> list[Chem.Mol]:
        """The distinct molecules visited, in the order first seen.

        Returns:
            One molecule per step in ``smiles_history``, each a snapshot with the
            conformer it had when recorded.
        """
        return list(self._history)

    def describe(self) -> str:
        """A compact markdown inventory of the current groups.

        Lists each group's id, name, fragment, role, and connections. Call
        :meth:`describe_group` for a group's atom positions, rings, and topology.
        A physicochemical profile and structure alerts are appended, and when the
        session has a receptor and the ligand is posed in 3D the predicted
        affinity is appended too, so all of it updates with every edit.
        """
        overview = describe_tree(self.tree)
        lines = [overview, profile_markdown(self.molecule())]
        components = self._pose_components()
        if components is not None:
            lines.append(_affinity_line(components))
        intra = self._intra_components()
        if intra is not None:
            lines.append(_internal_energy_line(intra))
        return "\n\n".join(lines)

    def describe_group(
        self, group_id: int, *, radius: int | None = None, use_matrix: bool = False
    ) -> str:
        """The atom positions, rings, and neighbourhood of one group.

        Args:
            group_id: Id of the group to detail.
            radius: Farthest bond distance the positions section describes. None
                (default) covers the group's fused ring system.
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

    def pose_sdf(self) -> str:
        """The current 3D pose as one SDF record.

        Returns:
            A V2000 molblock of the current molecule, with its conformer and
            explicit hydrogens, plus the ``$$$$`` terminator so the text is a
            complete SDF record.

        Raises:
            ValueError: If the ligand lacks 3D coordinates.
        """
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            raise ValueError("write_pose needs a ligand with 3D coordinates")
        return str(Chem.MolToMolBlock(self.molecule())) + "$$$$\n"

    def swap(self, group_id: int, group: str | Chem.Mol) -> None:
        """Replace a group's fragment with a new group.

        The new group must have the same number of ports as the group it replaces.
        Use the group's own port labels (from :meth:`describe_group`) to control
        which attachment sits where. To change a group and drop a substituent,
        :meth:`remove` that substituent leaf first to free its port, then swap.

        Args:
            group_id: Id of the group to replace.
            group: A curated group name or a SMILES/Mol with matching ports.

        Raises:
            ValueError: If the group cannot be parsed, or its ports do not match
                the group's; the message names the ports and how to proceed.
        """
        node = self.tree.node(group_id)
        branches = self._branches(node)
        anchor_label = (
            max(branches, key=lambda label: branches[label][1])
            if len(node.current.ports) > 1
            else None
        )
        posed = node.current.mol.GetNumConformers() > 0
        try:
            region = swap_region(
                node.current, _as_group(group), anchor_label=anchor_label
            )
        except ValueError as error:
            raise self._port_mismatch_error(group_id, error) from error
        with self._edit():
            sub_nodes = self._apply(node, region)
            if anchor_label is not None and posed:
                self._follow_arms(sub_nodes, branches, anchor_label)

    def _port_mismatch_error(self, group_id: int, error: ValueError) -> ValueError:
        """Enrich a port-mismatch swap error with the group's ports and next steps.

        A non-port error (an unparseable group) is returned unchanged.

        Args:
            group_id: Id of the group being swapped.
            error: The original error from :func:`swap_region`.

        Returns:
            A ``ValueError`` naming each port, its neighbour, and the two recovery
            routes; or ``error`` itself when the failure is not about ports.
        """
        if "port" not in str(error):
            return error
        labels = {node.id: node for node in annotate(self.tree).nodes}
        node = labels[group_id]
        ports = ", ".join(
            f"[{ref.port}*]->{ref.name or labels[ref.node_id].classification} "
            f"[{ref.node_id}]"
            for ref in sorted(node.neighbors, key=lambda ref: ref.port)
        )
        hint = (
            f"{error}. This group has {len(node.ports)} port(s): {ports}. Provide a "
            f"group with these port labels to keep every attachment"
        )
        leaves = sorted(
            (ref for ref in node.neighbors if labels[ref.node_id].role == "leaf"),
            key=lambda ref: ref.node_id,
        )
        if leaves:
            hint += (
                f", or remove a substituent leaf first (e.g. remove "
                f"{leaves[0].node_id}) to free its port, then swap"
            )
        return ValueError(f"{hint}.")

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
        with self._edit():
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
        with self._edit():
            self._apply(node, region)
        return position_id

    def remove(self, group_id: int) -> tuple[int, int]:
        """Delete a group and its subtree, capping the parent with hydrogen.

        Args:
            group_id: Id of the group to remove.

        Returns:
            The ``(group_id, position_id)`` of the capped attachment: the kept
            group and the hydrogen id where the removed group was attached, so it
            can be replaced with :meth:`grow`.

        Raises:
            ValueError: If the group is the root scaffold.
        """
        node = self.tree.node(group_id)
        with self._edit():
            capped = self.tree.remove_subtree(node)
        return capped

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

    def residues_near(
        self, group_id: int, position_id: int | None = None, cutoff: float = 4.5
    ) -> str:
        """List the receptor residues a group (or one of its atoms) contacts.

        Reports every residue with an atom within ``cutoff`` of the group's heavy
        atoms — or, when ``position_id`` is given, of that one atom — closest
        first. This is the group-to-residues view: what a substituent touches in
        the pocket. Unlike ``contacts``, which keeps only the closest atom per
        residue, it names every residue near the atom, so an atom-specific query
        ("which residues does this chlorine reach?") is answered in full.

        Args:
            group_id: Id of the group to measure from (from ``describe``).
            position_id: Optional heavy-atom id in the group (from
                ``describe_group``); omit to measure from the whole group.
            cutoff: Contact radius in angstrom.

        Returns:
            Markdown: a table of residue and its minimum distance to the target,
            sorted closest first.

        Raises:
            ValueError: If no receptor is loaded, the ligand lacks 3D coordinates,
                or ``position_id`` is not a heavy atom of the group.
        """
        if self.receptor is None:
            raise ValueError("spatial contacts need a receptor")
        mol = self.tree.node(group_id).current.mol
        if not mol.GetNumConformers():
            raise ValueError("spatial contacts need a ligand with 3D coordinates")
        target = _target_positions(mol, position_id)
        hits = [
            (residue, min_distance(target, self.receptor.residue_positions(residue)))
            for residue in self.receptor.residues()
        ]
        hits = [(residue, dist) for residue, dist in hits if dist <= cutoff]
        hits.sort(key=lambda hit: hit[1])
        return _residues_near_report(
            group_id, self._label(group_id), position_id, cutoff, hits
        )

    def matches(self, pattern: str) -> str:
        """Search the current molecule for a substructure, by name or SMILES/SMARTS.

        Check-your-work tool: after a ring swap, confirm you built the ring you
        meant. ``pattern`` is a heterocycle name (``"quinazoline"``) or a
        SMILES/SMARTS query. It reports whether the whole molecule contains the
        pattern and which single group contains it, and names the ring when the
        pattern is a known heterocycle — so building a quinazoline and searching
        ``"quinazoline"`` reports a match, while a quinoxaline would not.

        Args:
            pattern: A heterocycle name, or a SMILES/SMARTS substructure query.

        Returns:
            Markdown: the whole-molecule match count and the groups that contain it.

        Raises:
            ValueError: If ``pattern`` is neither a known name nor a parseable
                SMILES/SMARTS.
        """
        name: str | None = None
        named = heterocycle_smiles(pattern)
        if named is not None:
            query = Chem.MolFromSmiles(named)
            name = pattern.strip().lower()
        else:
            query = Chem.MolFromSmiles(pattern)
            if query is not None:
                name = heterocycle_name(pattern)
            else:
                query = Chem.MolFromSmarts(pattern)
        if query is None:
            raise ValueError(
                f"could not read pattern as a name, SMILES, or SMARTS: {pattern}"
            )
        whole = len(self.molecule().GetSubstructMatches(query, uniquify=True))
        group_hits = [
            (node.id, self._label(node.id))
            for node in self.tree.nodes
            if node.id is not None and node.current.mol.HasSubstructMatch(query)
        ]
        return _matches_report(pattern, name, whole, group_hits)

    def ring_change_note(self, before: Chem.Mol, after: Chem.Mol) -> str:
        """A note on a ring edit: the vendored ring and its ports' canonical positions.

        Used by the command layer to append position feedback to swap/grow/mutate
        results when the edit adds, removes, or changes a numbered ring — so the agent
        sees where each port landed (``[4*] at position 2``) without cross-referencing.

        Args:
            before: The edited group's fragment before the edit.
            after: The edited group's fragment after the edit.

        Returns:
            A leading-``; `` note, or ``""`` when no vendored ring was involved.
        """
        was, now = number_ring_system(before), number_ring_system(after)
        if now is None:
            if was is not None and _exact_ring(before, was):
                return f"; the {was[0]} ring was removed"
            return ""
        # Only name the previous ring when it was itself an exact vendored ring (not a
        # sub-ring of a larger fused system), and it differs from the new one.
        if was is not None and _exact_ring(before, was) and was[0] != now[0]:
            return (
                f"; {ring_ports_summary(before)} changed to {ring_ports_summary(after)}"
            )
        return f"; {ring_ports_summary(after)}"

    def minimize(
        self, group_id: int, degrees: float = 0.0, *, window: float = 180.0
    ) -> str:
        """Settle a group about its attachment bond to its best-scoring rotamer.

        The group turns about the single bond joining it to the rest of the
        molecule, carrying its own substituents as one rigid body. Every whole-
        degree turn is scored by the full Vinardo search energy — the ligand's
        intramolecular strain plus, when a receptor is loaded, the protein-ligand
        fit — and the best turn within ``window`` of the request is applied (ties
        toward the request). The constitution is unchanged, so the SMILES stays the
        same. The wide default window makes ``minimize(id)`` settle the group over
        the whole circle.

        Args:
            group_id: Id of the group to settle.
            degrees: Requested turn, in degrees; 0 settles from the current pose.
            window: Half-width, in degrees, of the search around ``degrees``.

        Returns:
            A short report: the applied turn and the search energy before and after.

        Raises:
            ValueError: If the ligand lacks 3D coordinates or the group is the
                only fragment.
        """
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            raise ValueError("minimize needs a ligand with 3D coordinates")
        node = self.tree.node(group_id)
        if not self.tree.neighbors(node):
            raise ValueError("cannot minimize the only group")
        axis_edge, _, moving = self._attachment(node)
        axis_point, axis_dir = self._axis(node, axis_edge)

        result = self._vinardo_scan(moving, axis_point, axis_dir, degrees, window)
        applied = result.window_best[0]
        with self._edit():
            matrix = axis_matrix(axis_point, axis_dir, math.radians(applied))
            for moving_node in moving:
                turned = Chem.Mol(moving_node.current.mol)
                rdMolTransforms.TransformConformer(turned.GetConformer(), matrix)
                moving_node.push(Fragment(turned))

        label = self._label(group_id)
        return _minimize_report(group_id, label, degrees, applied, result)

    def _vinardo_scan(
        self,
        moving: set[FragmentNode],
        axis_point: np.ndarray,
        axis_dir: np.ndarray,
        degrees: float,
        window: float,
    ) -> ScanResult:
        """Score every one-degree turn of the moving side by the Vinardo energy.

        The ligand typing, receptor data, intramolecular pair mask, and rotatable-
        bond count are fixed across the turn, so they are computed once; each turn
        rotates only the moving atoms and re-scores intermolecular + intramolecular
        Vinardo (``vinardo_objective``).
        """
        heavy = Chem.RemoveAllHs(self.molecule())
        coords = heavy.GetConformer().GetPositions()
        typing = atom_typing(heavy)
        intra_mask = intramolecular_mask(heavy)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(heavy, strict=False)
        node_id = np.array([a.GetIntProp("node_id") for a in heavy.GetAtoms()])
        moving_mask = np.isin(node_id, [n.id for n in moving])
        moving_coords = coords[moving_mask]
        prot_coords, prot_typing = (
            self.receptor.scoring_context()
            if self.receptor is not None
            else (None, None)
        )

        scores = np.empty(360)
        for degree in range(360):
            matrix = axis_matrix(axis_point, axis_dir, math.radians(degree))
            turned = coords.copy()
            turned[moving_mask] = moving_coords @ matrix[:3, :3].T + matrix[:3, 3]
            scores[degree] = vinardo_objective(
                turned, typing, prot_coords, prot_typing, intra_mask, n_rot
            )
        return scan_result(scores, target_deg=degrees, window_deg=window)

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

    def affinity(self, mol: Chem.Mol | None = None) -> float | None:
        """Predicted Vinardo affinity of a pose, or None without a posed receptor.

        Args:
            mol: The molecule to score. Defaults to the current molecule, so the
                trace can score each past step against the same receptor.

        Returns:
            The total Vinardo score (lower is a better fit) when the session has a
            receptor and the ligand is posed in 3D; None otherwise.
        """
        components = self._pose_components(mol)
        return None if components is None else components.total

    def internal_energy(self, mol: Chem.Mol | None = None) -> float | None:
        """The ligand's internal Vinardo energy, or None without 3D coordinates.

        This is the intramolecular energy (Vina's search term), not a binding
        score: it rises sharply when an edit leaves groups clashing inside the
        molecule, and needs no receptor. The reported affinity stays
        protein-ligand only.

        Args:
            mol: The molecule to score. Defaults to the current molecule.

        Returns:
            The intramolecular Vinardo total (lower is better) when the ligand is
            posed in 3D; None otherwise.
        """
        components = self._intra_components(mol)
        return None if components is None else components.total

    def _intra_components(self, mol: Chem.Mol | None = None) -> ScoreComponents | None:
        """The ligand's intramolecular score breakdown, or None without 3D coords."""
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            return None
        return score_intramolecular(self.molecule() if mol is None else mol)

    def _pose_components(self, mol: Chem.Mol | None = None) -> ScoreComponents | None:
        """A pose's Vinardo score, or None without a posed receptor.

        Args:
            mol: The molecule to score. Defaults to the current molecule.

        Returns:
            The score breakdown when the session has a receptor and the ligand is
            posed in 3D; None otherwise, so callers can omit affinity cleanly.
        """
        if self.receptor is None:
            return None
        if not self.tree.nodes[0].current.mol.GetNumConformers():
            return None
        coords, typing = self.receptor.scoring_context()
        return score_pose(self.molecule() if mol is None else mol, coords, typing)

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

    def _branches(self, node: FragmentNode) -> dict[int, tuple[set[FragmentNode], int]]:
        """Each of a node's ports mapped to the branch beyond it and its size.

        Args:
            node: The node whose incident branches to measure.

        Returns:
            Port label -> (the nodes on that branch, its heavy-atom count).
        """
        result: dict[int, tuple[set[FragmentNode], int]] = {}
        for edge, other in self.tree.neighbors(node):
            branch = self._moving_side(other, edge)
            size = sum(heavy_count(n.current.mol) for n in branch)
            result[edge.label] = (branch, size)
        return result

    def _follow_arms(
        self,
        placed: list[FragmentNode],
        branches: dict[int, tuple[set[FragmentNode], int]],
        anchor_label: int,
    ) -> None:
        """Move each non-anchor branch to reconnect to the newly placed group.

        The anchor branch is the fixed frame and does not move. Every other branch
        is rigidly transformed so its attachment port lands on the placed group's
        matching port, keeping the branch's own internal geometry.

        Args:
            placed: The sub-nodes the swapped region became.
            branches: The swapped node's branches by port label (from _branches).
            anchor_label: The pinned port; its branch stays put.
        """
        for label, (branch, _size) in branches.items():
            if label == anchor_label:
                continue
            new_anchor, new_dummy = self._port_frame(placed, label)
            # The arm's bonding atom sits a bond length out along the port, not at
            # the placed dummy (an embedded dummy can land too close to its anchor).
            direction = new_dummy - new_anchor
            target = new_anchor + _BOND_LENGTH * direction / np.linalg.norm(direction)
            arm_anchor, arm_dummy = self._port_frame(branch, label)
            matrix = align_transform(
                arm_anchor, arm_dummy - arm_anchor, target, new_anchor - target
            )
            for arm_node in branch:
                moved = Chem.Mol(arm_node.current.mol)
                rdMolTransforms.TransformConformer(moved.GetConformer(), matrix)
                arm_node.push(Fragment(moved))

    def _port_frame(
        self, nodes: Iterable[FragmentNode], label: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The (anchor, dummy) coordinates of the port with ``label`` among nodes."""
        node = next(n for n in nodes if any(p.label == label for p in n.current.ports))
        port = next(p for p in node.current.ports if p.label == label)
        conf = node.current.mol.GetConformer()
        anchor = np.array(list(conf.GetAtomPosition(port.anchor_idx)))
        dummy = np.array(list(conf.GetAtomPosition(port.dummy_idx)))
        return anchor, dummy

    def _axis(
        self, node: FragmentNode, axis_edge: Edge
    ) -> tuple[np.ndarray, np.ndarray]:
        """The rotation axis: a point on it and its direction, from ``node``'s port."""
        port = next(p for p in node.current.ports if p.label == axis_edge.label)
        conf = node.current.mol.GetConformer()
        point = np.array(list(conf.GetAtomPosition(port.anchor_idx)))
        direction = np.array(list(conf.GetAtomPosition(port.dummy_idx))) - point
        return point, direction

    def _label(self, group_id: int) -> str:
        """The chemist-facing name of a group, for reports."""
        return _name({n.id: n for n in annotate(self.tree).nodes}[group_id])

    def _apply(self, node: FragmentNode, region_mol: Chem.Mol) -> list[FragmentNode]:
        """Resplice an edited region into the tree; return the new sub-nodes.

        Callers wrap this in :meth:`_edit`, which snapshots the tree for undo and
        records the result.
        """
        sub_nodes = self.tree.resplice(node, region_mol)
        self._normalize_new_hydrogens(sub_nodes)
        return sub_nodes

    @contextlib.contextmanager
    def _edit(self) -> Iterator[None]:
        """Run a tree edit atomically, snapshotting the tree for undo.

        Snapshots the current tree before the edit. If the edit raises, the tree
        is restored to that snapshot and the error re-raised; on success the new
        molecule is recorded in the history.
        """
        snapshot = self.tree.copy()
        self._snapshots.append(snapshot)
        try:
            yield
        except Exception:
            self._snapshots.pop()
            self.tree = snapshot
            raise
        self._record()

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
        if not self._snapshots:
            raise ValueError("nothing to undo")
        self.tree = self._snapshots.pop()
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

# Single-bond length (angstrom) used to reconnect a moved branch to a swapped group.
_BOND_LENGTH = 1.5


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


def _internal_energy_line(components: ScoreComponents) -> str:
    """The ligand internal-energy line appended to a 3D describe."""
    return (
        f"**Internal energy (Vinardo):** {components.total:.2f} "
        "(intramolecular; lower is better)"
    )


def _clashes_report(findings: list[tuple[float, str]]) -> str:
    """Render the clash findings, already sorted worst first."""
    if not findings:
        return "# Clashes\n\nNo clashes."
    return "\n".join(["# Clashes", ""] + [f"- {desc}" for _, desc in findings])


def _minimize_report(
    group_id: int, label: str, requested: float, applied: int, result: ScanResult
) -> str:
    """Render the outcome of a Vinardo torsion settle for the agent."""
    before, after = result.current, result.window_best[1]
    lines = [
        f"Settled [{group_id}] {label} by {applied} deg (requested {requested:g} deg).",
        f"Energy {before:.2f} -> {after:.2f}.",
    ]
    global_deg, global_score = result.global_best
    if global_score < after - 0.05:
        lines.append(
            f"A larger turn to {global_deg} deg would lower the energy to "
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


def _target_positions(mol: Chem.Mol, position_id: int | None) -> np.ndarray:
    """Coordinates to measure from: one heavy atom, or all of a group's heavy atoms.

    Args:
        mol: The group's fragment molecule, posed in 3D.
        position_id: A heavy-atom id to measure from, or None for the whole group.

    Returns:
        An (N, 3) array of the target coordinates.

    Raises:
        ValueError: If ``position_id`` is out of range or not a heavy atom.
    """
    conf = mol.GetConformer()
    if position_id is None:
        idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    else:
        if not 0 <= position_id < mol.GetNumAtoms():
            raise ValueError(f"no atom {position_id} in this group")
        if mol.GetAtomWithIdx(position_id).GetAtomicNum() <= 1:
            raise ValueError(f"atom {position_id} is not a heavy atom")
        idxs = [position_id]
    return np.array([list(conf.GetAtomPosition(i)) for i in idxs])


def _residues_near_report(
    group_id: int,
    label: str,
    position_id: int | None,
    cutoff: float,
    hits: list[tuple[Residue, float]],
) -> str:
    """Render the residues-near table, hits already sorted closest first."""
    target = f"[{group_id}] {label}"
    if position_id is not None:
        target += f" atom {position_id}"
    lines = [f"# Residues within {cutoff:g} A of {target}", ""]
    if not hits:
        lines.append(f"No residue within {cutoff:g} A.")
        return "\n".join(lines)
    lines += ["| residue | distance (A) |", "|---|---|"]
    for residue, distance in hits:
        lines.append(f"| {residue.name}{residue.number} | {distance:.2f} |")
    return "\n".join(lines)


def _exact_ring(mol: Chem.Mol, numbered: tuple[str, dict[int, int]] | None) -> bool:
    """True if a vendored ring matched all of ``mol``'s ring system, not a sub-ring."""
    if numbered is None:
        return False
    ring_atoms = sum(1 for a in mol.GetAtoms() if a.IsInRing())
    return bool(
        ring_atoms == Chem.MolFromSmiles(HETEROCYCLES[numbered[0]]).GetNumAtoms()
    )


def _matches_report(
    pattern: str, name: str | None, whole: int, group_hits: list[tuple[int, str]]
) -> str:
    """Render the substructure-search report; ``name`` labels a known ring."""
    header = f"# Substructure `{pattern}`"
    if name:
        header += f" ({name})"
    lines = [header, ""]
    if whole == 0:
        lines.append("Whole molecule: no match.")
        return "\n".join(lines)
    lines.append(f"Whole molecule: {whole} {'match' if whole == 1 else 'matches'}.")
    if group_hits:
        groups = ", ".join(f"[{gid}] {label}" for gid, label in group_hits)
        lines.append(f"Contained in group(s): {groups}.")
    else:
        lines.append("Not contained in any single group.")
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
        number = Chem.GetPeriodicTable().GetAtomicNumber(element.strip())
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
