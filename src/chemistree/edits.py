"""Fragment edits.

``swap`` replaces a fragment with a new group, keeping its attachment ports and
overlaying the new group onto the retained 3D frame: atoms shared with the old
fragment (by MCS) plus the port anchors are fixed to their original positions and
the rest is relaxed with UFF.
"""

from __future__ import annotations

from collections.abc import Iterable

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdFMCS, rdMolAlign

from chemistree.fragment import Fragment
from chemistree.geometry import chiral_volume
from chemistree.tree import FragmentNode

_EMBED_SEED = 0xF00D  # fixed so placement is reproducible


def swap(node: FragmentNode, group: str | Chem.Mol) -> None:
    """Replace a node's current fragment with a new group.

    The new group must provide the same attachment ports as the fragment it
    replaces, matched by label; a lone port is mapped automatically. When the
    fragment carries 3D coordinates the new group is embedded and overlaid onto
    the retained frame, preserving the geometry of the untouched molecule.

    Args:
        node: Node whose current fragment is replaced.
        group: New group as a SMILES string or Mol, with one dummy atom per port.

    Raises:
        ValueError: If the group cannot be parsed, has no ports, or its port
            labels do not match the fragment's.
    """
    node.push(Fragment(swap_region(node.current, group)))


def swap_region(old: Fragment, group: str | Chem.Mol) -> Chem.Mol:
    """Build the replacement mol for a swap, placed onto the old fragment's frame.

    This is ``swap`` without the tree side effect: it parses the group, matches its
    ports to the old fragment's labels, and (when the old fragment has 3D coords)
    overlays it onto the retained frame. The caller decides how to graft the result
    into the tree.

    Args:
        old: Fragment being replaced, whose ports and frame are preserved.
        group: New group as a SMILES string or Mol, with one dummy per port.

    Returns:
        The new group mol, port-labeled and (when applicable) placed in 3D.
    """
    new = _parse_group(group)
    _assign_port_labels(old, new)
    if old.mol.GetNumConformers():
        new = _place(old, new)
    return new


def grow_region(old: Fragment, hydrogen: int, group: str | Chem.Mol) -> Chem.Mol:
    """Build the edited region for growing a group where a hydrogen was.

    The hydrogen at ``hydrogen`` is removed and the group is bonded to its heavy
    parent, so the group takes that hydrogen's place; the old fragment's ports and
    frame are preserved. The caller re-fragments the result, so whether the group
    becomes its own node or merges follows the same policy as construction.

    Args:
        old: Fragment being grown, with explicit hydrogens.
        hydrogen: Atom id of the hydrogen to replace (its parent is the grow site).
        group: New group as a SMILES string or Mol, with one dummy attachment.

    Returns:
        The augmented region mol, placed in 3D when the fragment has coords.

    Raises:
        ValueError: If ``hydrogen`` is not a hydrogen atom, or the group has no port.
    """
    atom = old.mol.GetAtomWithIdx(hydrogen)
    if atom.GetAtomicNum() != 1:
        raise ValueError(f"atom {hydrogen} is not a hydrogen; grow replaces a hydrogen")
    heavy = atom.GetNeighbors()[0].GetIdx()
    augmented = _attach_at_hydrogen(old.mol, hydrogen, heavy, _parse_group(group))
    if old.mol.GetNumConformers():
        augmented = _place(old, augmented)
    return augmented


def mutate_region(old: Fragment, atom: int, element: int) -> Chem.Mol:
    """Build the edited region for mutating one heavy atom's element.

    Args:
        old: Fragment being edited.
        atom: Heavy-atom id whose element changes.
        element: Atomic number of the new element.

    Returns:
        The mutated region mol, with explicit hydrogens, placed in 3D when the
        fragment has coords.

    Raises:
        ValueError: If the change leaves an invalid valence.
    """
    new = _mutated_mol(old.mol, atom, element)
    if old.mol.GetNumConformers():
        return _place(old, new)
    return Chem.AddHs(new)


def _attach_at_hydrogen(
    scaffold: Chem.Mol, hydrogen: int, heavy: int, group: Chem.Mol
) -> Chem.Mol:
    """Bond a group's anchor to ``heavy``, removing the specific ``hydrogen`` there."""
    rw = Chem.RWMol(Chem.CombineMols(scaffold, group))
    offset = scaffold.GetNumAtoms()
    g_dummy = next(
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetIdx() >= offset and a.GetAtomicNum() == 0
    )
    g_anchor = rw.GetAtomWithIdx(g_dummy).GetNeighbors()[0].GetIdx()
    rw.AddBond(heavy, g_anchor, Chem.BondType.SINGLE)
    for idx in sorted([g_dummy, hydrogen], reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def mutate_atom(node: FragmentNode, atom: int, element: int) -> None:
    """Change one atom's element in a node's fragment (e.g. aromatic C to N).

    The atom is addressed by its index in the fragment's heavy-atom skeleton.
    Ports are untouched. When the fragment carries 3D coordinates the mutated
    fragment is re-placed by MCS overlay, so the rest of the ring keeps its frame.
    This is the atom-level path to ring heteroatom edits: turning the ring CH
    between two substituents into N converts a benzene into the matching pyridine.

    Args:
        node: Node whose current fragment is edited.
        atom: Heavy-atom index of the atom to change.
        element: Atomic number of the new element.

    Raises:
        ValueError: If the change leaves an invalid valence.
    """
    old = node.current
    new = _mutated_mol(old.mol, atom, element)
    if old.mol.GetNumConformers():
        new = _place(old, new)
    node.push(Fragment(new))


def _mutated_mol(mol: Chem.Mol, atom: int, element: int) -> Chem.Mol:
    """Return a copy of ``mol`` with ``atom``'s element set to ``element``.

    Works on the heavy-atom skeleton (implicit Hs) so the H count is recomputed
    for the new element, and re-sanitizes to re-perceive aromaticity.

    Raises:
        ValueError: If sanitization fails (an impossible valence).
    """
    heavy = Chem.RWMol(Chem.RemoveHs(mol))
    target = heavy.GetAtomWithIdx(atom)
    was_aromatic = target.GetIsAromatic()
    old_symbol = target.GetSymbol()
    new_symbol = Chem.GetPeriodicTable().GetElementSymbol(element)
    target.SetAtomicNum(element)
    target.SetFormalCharge(0)
    target.SetNumExplicitHs(0)
    target.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(heavy)
    except rdchem.MolSanitizeException as error:
        ring = "aromatic " if was_aromatic else ""
        raise ValueError(
            f"cannot mutate {ring}{old_symbol} at atom {atom} to {new_symbol}: "
            f"the result has an invalid valence ({error})"
        ) from error
    return heavy.GetMol()


def add_substituent(node: FragmentNode, atom: int, group: str | Chem.Mol) -> None:
    """Grow a group at an open atom of the node's fragment.

    The group is attached at ``atom`` (replacing one hydrogen) and the node is
    swapped for the augmented fragment, so its ports are preserved and the new
    group is placed with the same MCS-overlay + UFF machinery as ``swap``.

    Args:
        node: Node whose fragment grows the group.
        atom: Index (in the node's current fragment) of the atom to grow from.
        group: New group as a SMILES string or Mol, with one dummy attachment.

    Raises:
        ValueError: If the group cannot be parsed or has no port.
    """
    augmented = _attach(node.current.mol, atom, _parse_group(group))
    swap(node, augmented)


def _attach(scaffold: Chem.Mol, atom: int, group: Chem.Mol) -> Chem.Mol:
    """Bond a group's anchor to a scaffold atom, consuming one hydrogen there."""
    hydrogens = scaffold.GetAtomWithIdx(atom).GetTotalNumHs()
    rw = Chem.RWMol(Chem.CombineMols(scaffold, group))

    offset = scaffold.GetNumAtoms()
    g_dummy = next(
        a.GetIdx()
        for a in rw.GetAtoms()
        if a.GetIdx() >= offset and a.GetAtomicNum() == 0
    )
    g_anchor = rw.GetAtomWithIdx(g_dummy).GetNeighbors()[0].GetIdx()
    rw.AddBond(atom, g_anchor, Chem.BondType.SINGLE)

    target = rw.GetAtomWithIdx(atom)
    explicit_h = next(
        (n.GetIdx() for n in target.GetNeighbors() if n.GetAtomicNum() == 1), None
    )
    remove = [g_dummy]
    if explicit_h is not None:
        remove.append(explicit_h)
    else:
        target.SetNumExplicitHs(max(0, hydrogens - 1))
        target.SetNoImplicit(True)
    for idx in sorted(remove, reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _parse_group(group: str | Chem.Mol) -> Chem.Mol:
    """Parse a swap group (SMILES or Mol) and require at least one port.

    Raises:
        ValueError: If it cannot be parsed or has no dummy attachment.
    """
    mol = Chem.MolFromSmiles(group) if isinstance(group, str) else Chem.Mol(group)
    if mol is None:
        raise ValueError(f"could not parse group: {group!r}")
    if not _dummies(mol):
        raise ValueError("group must have at least one dummy attachment (*)")
    return mol


def _dummies(mol: Chem.Mol) -> list[Chem.Atom]:
    """The dummy (port) atoms of a molecule."""
    return [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _assign_port_labels(old: Fragment, new: Chem.Mol) -> None:
    """Match the new group's dummies to the old fragment's port labels.

    Explicit labels that already match are honored. Bare dummies (isotope 0) are
    reconciled: the fragment's labels are assigned to them in atom order, so a
    caller can write ``[*]O[*]`` to replace a 2-port ``-NH-`` linker without
    knowing the internal port numbers.

    Args:
        old: Fragment being replaced, whose port labels must be preserved.
        new: Replacement group; its dummies are labeled in place.

    Raises:
        ValueError: If the port counts differ, or the dummies carry explicit
            labels that do not match the fragment's.
    """
    old_labels = sorted(p.label for p in old.ports)
    dummies = _dummies(new)
    if len(dummies) != len(old_labels):
        raise ValueError(
            f"group has {len(dummies)} port(s); fragment has {len(old_labels)}"
        )
    isotopes = [a.GetIsotope() for a in dummies]
    if sorted(isotopes) == old_labels:
        return  # explicit labels already match
    if all(isotope == 0 for isotope in isotopes):
        for dummy, label in zip(dummies, old_labels):
            dummy.SetIsotope(label)
        return
    raise ValueError("group port labels must match the fragment's port labels")


def _place(old: Fragment, new: Chem.Mol) -> Chem.Mol:
    """Embed the new group and overlay it onto the old fragment's frame."""
    new = Chem.AddHs(new)
    AllChem.EmbedMolecule(new, randomSeed=_EMBED_SEED)

    correspondence = _mcs_correspondence(old.mol, new)
    fixed, anchors = _fixed_atoms(old, new, correspondence)
    _free_inverted_centers(old, new, correspondence, anchors, fixed)

    old_conf = old.mol.GetConformer()
    rdMolAlign.AlignMol(new, old.mol, atomMap=[(ni, oi) for ni, oi in fixed.items()])
    conf = new.GetConformer()
    corr_new_to_old = {ni: oi for oi, ni in correspondence}
    pinned = {
        ni: oi
        for ni, oi in fixed.items()
        if _should_pin(new, ni, oi, old.mol, corr_new_to_old)
    }
    for ni, oi in pinned.items():
        conf.SetAtomPosition(ni, old_conf.GetAtomPosition(oi))

    _relax(new, pinned)
    return new


def _should_pin(
    new: Chem.Mol,
    ni: int,
    oi: int,
    old: Chem.Mol,
    corr_new_to_old: dict[int, int],
) -> bool:
    """Whether to fix a matched atom to its old position.

    Heavy atoms and port dummies are always pinned; they define the frame. A
    hydrogen is pinned only when its parent heavy atom is itself a matched pair.
    A genuinely corresponding hydrogen (on a preserved stereocenter) is then held
    so the center keeps its handedness, while a hydrogen matched by coincidence
    (a new ring H mapped to an old methyl H) is left free to relax, so the new
    ring is not dragged onto a wrong position.

    Args:
        new: The new group being placed.
        ni: Index in ``new`` of the matched atom.
        oi: Index in ``old`` the atom is matched to.
        old: The old fragment molecule.
        corr_new_to_old: Map of matched new-atom index to old-atom index.

    Returns:
        True if the atom should be pinned to its old position.
    """
    atom = new.GetAtomWithIdx(ni)
    if atom.GetAtomicNum() != 1:
        return True
    new_parent = atom.GetNeighbors()[0].GetIdx()
    old_parent = old.GetAtomWithIdx(oi).GetNeighbors()[0].GetIdx()
    return bool(corr_new_to_old.get(new_parent) == old_parent)


def _fixed_atoms(
    old: Fragment, new: Chem.Mol, correspondence: list[tuple[int, int]]
) -> tuple[dict[int, int], set[int]]:
    """Map new-group atom indices to the old positions they overlay.

    Shared atoms come from the MCS; the port anchors and dummies are always
    pinned so the reconnection geometry is preserved. Returns the fixed map and
    the set of port-anchor indices (which stereo handling must never free).
    """
    fixed = {ni: oi for oi, ni in correspondence}
    anchors = set()
    for port in old.ports:
        dummy = next(
            a
            for a in new.GetAtoms()
            if a.GetAtomicNum() == 0 and a.GetIsotope() == port.label
        )
        anchor = dummy.GetNeighbors()[0].GetIdx()
        fixed[dummy.GetIdx()] = port.dummy_idx
        fixed[anchor] = port.anchor_idx
        anchors.add(anchor)
    return fixed, anchors


def _free_inverted_centers(
    old: Fragment,
    new: Chem.Mol,
    correspondence: list[tuple[int, int]],
    anchors: set[int],
    fixed: dict[int, int],
) -> None:
    """Release matched stereocenters the new group explicitly inverts.

    Stereo is conserved by default: fixing a matched center to its old
    coordinates preserves its chiral volume. When the new group instead states
    the opposite configuration at a matched center, that center (and its
    non-anchor matched neighbors) is dropped from the fixed set so the embedded
    geometry stands. Port anchors are always conserved.
    """
    old_conf = old.mol.GetConformer()
    new_conf = new.GetConformer()
    old_of_new = {ni: oi for oi, ni in correspondence}
    for oi, ni in correspondence:
        atom = new.GetAtomWithIdx(ni)
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED or ni in anchors:
            continue
        matched = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in old_of_new]
        if len(matched) < 3:
            continue
        n3 = matched[:3]
        o3 = [old_of_new[x] for x in n3]
        stated = chiral_volume(new_conf, ni, *n3)
        original = chiral_volume(old_conf, oi, *o3)
        if stated * original < 0:
            fixed.pop(ni, None)
            for x in matched:
                if x not in anchors:
                    fixed.pop(x, None)


def _mcs_correspondence(a: Chem.Mol, b: Chem.Mol) -> list[tuple[int, int]]:
    """Atom-index pairs (a, b) shared by the two molecules' MCS.

    Atoms match across element (``CompareAnyHeavyAtom``) so a ring survives a
    heteroatom change: swapping a benzene for a pyridine still overlays the whole
    ring onto the old frame. With strict element matching the ring would not
    match, leaving it pinned by only its two port anchors, which warps the placed
    geometry. Bonds must still match order, and rings only match complete rings,
    so the overlay stays structurally faithful.
    """
    mcs = rdFMCS.FindMCS(
        [a, b],
        atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    if not mcs.smartsString:
        return []
    patt = Chem.MolFromSmarts(mcs.smartsString)
    pairs = zip(a.GetSubstructMatch(patt), b.GetSubstructMatch(patt))
    # Drop cross-kind pairs. A single-atom MCS pattern can be a wildcard that
    # matches a heavy atom in one molecule and a hydrogen in the other; such a
    # pair is meaningless and skews the overlay. Keep only pairs whose atoms are
    # the same kind (heavy, hydrogen, or dummy). Hydrogen pairs still guide the
    # alignment, but ``_place`` never pins a hydrogen to an exact position.
    return [
        (oi, ni)
        for oi, ni in pairs
        if _atom_kind(a.GetAtomWithIdx(oi)) == _atom_kind(b.GetAtomWithIdx(ni))
    ]


def _atom_kind(atom: Chem.Atom) -> str:
    """Classify an atom as ``heavy``, ``hydrogen``, or ``dummy`` for matching."""
    number = atom.GetAtomicNum()
    if number == 0:
        return "dummy"
    return "hydrogen" if number == 1 else "heavy"


def _relax(mol: Chem.Mol, fixed: Iterable[int]) -> None:
    """UFF-minimize the free atoms, holding the fixed ones in place."""
    fixed = set(fixed)
    if all(a.GetIdx() in fixed for a in mol.GetAtoms()):
        return  # nothing free to move

    # UFF cannot type dummy atoms; stand in a hydrogen (fixed) during minimization.
    dummies = [
        (a.GetIdx(), a.GetIsotope()) for a in mol.GetAtoms() if a.GetAtomicNum() == 0
    ]
    for idx, _ in dummies:
        mol.GetAtomWithIdx(idx).SetAtomicNum(1)
    mol.UpdatePropertyCache(strict=False)

    ff = AllChem.UFFGetMoleculeForceField(mol)
    for idx in fixed:
        ff.AddFixedPoint(idx)
    ff.Minimize(maxIts=1000)

    for idx, iso in dummies:
        atom = mol.GetAtomWithIdx(idx)
        atom.SetAtomicNum(0)
        atom.SetIsotope(iso)
