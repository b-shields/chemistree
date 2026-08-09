"""Fragment edits.

``swap`` replaces a fragment with a new group, keeping its attachment ports and
overlaying the new group onto the retained 3D frame: atoms shared with the old
fragment (by MCS) plus the port anchors are fixed to their original positions and
the rest is relaxed with UFF.
"""

from __future__ import annotations

from collections.abc import Iterable

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign

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
    old = node.current
    new = _parse_group(group)
    _assign_port_labels(old, new)
    if old.mol.GetNumConformers():
        new = _place(old, new)
    node.push(Fragment(new))


def _parse_group(group: str | Chem.Mol) -> Chem.Mol:
    mol = Chem.MolFromSmiles(group) if isinstance(group, str) else Chem.Mol(group)
    if mol is None:
        raise ValueError(f"could not parse group: {group!r}")
    if not _dummies(mol):
        raise ValueError("group must have at least one dummy attachment (*)")
    return mol


def _dummies(mol: Chem.Mol) -> list[Chem.Atom]:
    return [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _assign_port_labels(old: Fragment, new: Chem.Mol) -> None:
    """Match the new group's dummies to the old fragment's port labels."""
    old_labels = sorted(p.label for p in old.ports)
    dummies = _dummies(new)
    if len(dummies) != len(old_labels):
        raise ValueError(
            f"group has {len(dummies)} port(s); fragment has {len(old_labels)}"
        )
    if len(old_labels) == 1:
        dummies[0].SetIsotope(old_labels[0])  # unambiguous
    elif sorted(a.GetIsotope() for a in dummies) != old_labels:
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
    for ni, oi in fixed.items():
        conf.SetAtomPosition(ni, old_conf.GetAtomPosition(oi))

    _relax(new, fixed)
    return new


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
    """Atom-index pairs (a, b) shared by the two molecules' MCS."""
    mcs = rdFMCS.FindMCS(
        [a, b],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    if not mcs.smartsString:
        return []
    patt = Chem.MolFromSmarts(mcs.smartsString)
    return list(zip(a.GetSubstructMatch(patt), b.GetSubstructMatch(patt)))


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
