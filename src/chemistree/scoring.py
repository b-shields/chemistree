"""Vinardo empirical scoring for a ligand pose against a receptor.

Scores a posed ligand where it sits: a single evaluation of the Vinardo empirical
function, no docking search. The function is the one from Quiroga & Villarreal
(2016) as implemented in smina, summed over ligand/receptor heavy-atom pairs:

    score = (w_gauss1 * Gauss1 + w_repulsion * Repulsion
             + w_hydrophobic * Hydrophobic + w_hbond * HBond)
            / (1 + w_rot * N_rot)

Lower (more negative) scores mean a better fit. The math is vendored from
``cmxflow`` (MIT) and trimmed to the scoring path: heavy-atom only, dense
numpy over all pairs, no gradients or pose optimization.

Reference:
    Quiroga & Villarreal (2016). Vinardo: A Scoring Function Based on AutoDock
    Vina Improves Scoring, Docking, and Virtual Screening. PLOS ONE 11(5):
    e0155183. https://doi.org/10.1371/journal.pone.0155183
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@dataclass(frozen=True)
class EmpiricalParams:
    """Vinardo scoring weights and cutoffs.

    Defaults are the smina Vinardo values.

    Attributes:
        w_gauss1: Weight of the Gaussian attractive term.
        w_repulsion: Weight of the repulsion term.
        w_hydrophobic: Weight of the hydrophobic term.
        w_hbond: Weight of the hydrogen-bond term.
        w_rot: Torsional-entropy weight. The score is divided by
            ``1 + w_rot * N_rot`` for ``N_rot`` ligand rotatable bonds.
        gauss1_offset: Gaussian center, on surface distance.
        gauss1_width: Gaussian width.
        hydro_good: Inner surface-distance cutoff of the hydrophobic ramp.
        hydro_bad: Outer surface-distance cutoff of the hydrophobic ramp.
        hbond_good: Inner surface-distance cutoff of the hydrogen-bond ramp.
    """

    w_gauss1: float = -0.045
    w_repulsion: float = 0.800
    w_hydrophobic: float = -0.035
    w_hbond: float = -0.600
    w_rot: float = 0.02
    gauss1_offset: float = 0.0
    gauss1_width: float = 0.8
    hydro_good: float = 0.0
    hydro_bad: float = 2.5
    hbond_good: float = -0.6


@dataclass(frozen=True)
class ScoreComponents:
    """The four Vinardo terms of one pose, weighted and totaled.

    Raw fields are the unweighted pair sums. Each property applies the term's
    weight and the torsion divisor, so ``total`` is the reported score.

    Attributes:
        gauss1_raw: Unweighted Gaussian attractive sum.
        repulsion_raw: Unweighted repulsion sum.
        hydrophobic_raw: Unweighted hydrophobic sum.
        hbond_raw: Unweighted hydrogen-bond sum.
        params: The weights and cutoffs used.
        n_rot: Ligand rotatable-bond count (the torsion divisor's input).
    """

    gauss1_raw: float
    repulsion_raw: float
    hydrophobic_raw: float
    hbond_raw: float
    params: EmpiricalParams
    n_rot: int

    @property
    def _divisor(self) -> float:
        """The torsional-entropy divisor ``1 + w_rot * N_rot``."""
        return 1.0 + self.params.w_rot * self.n_rot

    @property
    def gauss1(self) -> float:
        """Weighted Gaussian attractive term."""
        return self.params.w_gauss1 * self.gauss1_raw / self._divisor

    @property
    def repulsion(self) -> float:
        """Weighted repulsion term."""
        return self.params.w_repulsion * self.repulsion_raw / self._divisor

    @property
    def hydrophobic(self) -> float:
        """Weighted hydrophobic term."""
        return self.params.w_hydrophobic * self.hydrophobic_raw / self._divisor

    @property
    def hbond(self) -> float:
        """Weighted hydrogen-bond term."""
        return self.params.w_hbond * self.hbond_raw / self._divisor

    @property
    def total(self) -> float:
        """The Vinardo pose score (lower is better)."""
        return self.gauss1 + self.repulsion + self.hydrophobic + self.hbond


# Vinardo heavy-atom radii (angstrom); aromatic carbon and any other element
# fall back to the two constants below.
_RADII: dict[int, float] = {
    6: 2.0,  # aliphatic carbon
    7: 1.7,  # nitrogen
    8: 1.6,  # oxygen
    9: 1.5,  # fluorine
    15: 2.1,  # phosphorus
    16: 2.0,  # sulfur
    17: 1.8,  # chlorine
    35: 2.0,  # bromine
    53: 2.2,  # iodine
}
_AROMATIC_CARBON_RADIUS = 1.9
_DEFAULT_RADIUS = 1.7

# smina/Vina XS atom typing as SMARTS. Hydrophobic: aromatic carbon, aliphatic
# carbon not bonded to a polar atom, or a halogen.
_HYDROPHOBIC = "[$([#6;a]),$([#6;A;!$([#6]~[#7,#8,#15,#16])]),$([#9,#17,#35,#53])]"
_HBOND_DONOR = (
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O;H1;+0]),$([n;H1;+0]),$([n;!H0;+1])"
    ",Li+1,Na+1,K+1,Cs+1,Mg+2,Ca+2,Mn+2,Zn+2]"
)
_HBOND_ACCEPTOR = (
    "[$([O;H1;v2]-[!$(*=[O,N,P,S])]),$([O;H0;v2]),$([O;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S]);!$(N-c)]),$([nH0,o;+0])]"
)


@dataclass(frozen=True)
class AtomTyping:
    """Per-atom Vinardo typing, aligned to a molecule's atom order.

    Attributes:
        radii: (N,) van der Waals radii.
        hydrophobic: (N,) mask of hydrophobic atoms.
        donor: (N,) mask of hydrogen-bond donors.
        acceptor: (N,) mask of hydrogen-bond acceptors.
    """

    radii: np.ndarray
    hydrophobic: np.ndarray
    donor: np.ndarray
    acceptor: np.ndarray


def atom_typing(mol: Chem.Mol) -> AtomTyping:
    """Assign Vinardo radii and hydrophobic/donor/acceptor masks to every atom.

    Args:
        mol: A molecule whose atoms are typed in place (typically heavy-atom only).

    Returns:
        The per-atom typing, aligned to ``mol``'s atom order.
    """
    return AtomTyping(
        radii=_atom_radii(mol),
        hydrophobic=_smarts_mask(mol, _HYDROPHOBIC),
        donor=_smarts_mask(mol, _HBOND_DONOR),
        acceptor=_smarts_mask(mol, _HBOND_ACCEPTOR),
    )


def _atom_radii(mol: Chem.Mol) -> np.ndarray:
    """Vinardo van der Waals radius of each atom, in molecule order."""
    radii = np.empty(mol.GetNumAtoms(), dtype=np.float64)
    for i, atom in enumerate(mol.GetAtoms()):
        number = atom.GetAtomicNum()
        if number == 6 and atom.GetIsAromatic():
            radii[i] = _AROMATIC_CARBON_RADIUS
        else:
            radii[i] = _RADII.get(number, _DEFAULT_RADIUS)
    return radii


def _smarts_mask(mol: Chem.Mol, smarts: str) -> np.ndarray:
    """Boolean mask of atoms matching a SMARTS pattern.

    Args:
        mol: The molecule to search.
        smarts: The SMARTS pattern.

    Returns:
        An (N,) boolean array, True where an atom is in any match.
    """
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return mask
    # maxMatches defaults to 1000, which silently drops matches on a protein with
    # thousands of atoms; raise it to the atom count so nothing is truncated.
    for match in mol.GetSubstructMatches(pattern, maxMatches=mol.GetNumAtoms()):
        for idx in match:
            mask[idx] = True
    return mask


def score_pose(
    ligand: Chem.Mol,
    protein_coords: np.ndarray,
    protein_typing: AtomTyping,
    params: EmpiricalParams | None = None,
) -> ScoreComponents:
    """Score a ligand pose against a receptor with the Vinardo function.

    The ligand is scored as it sits; hydrogens are stripped internally. The
    receptor is passed pre-typed (its geometry is fixed, so a caller types it
    once and reuses it).

    Args:
        ligand: The ligand molecule with a 3D conformer.
        protein_coords: (M, 3) receptor heavy-atom coordinates.
        protein_typing: The receptor's Vinardo typing, aligned to
            ``protein_coords``.
        params: Scoring weights and cutoffs; the smina Vinardo defaults if None.

    Returns:
        The four term sums with the weights and rotatable-bond count, whose
        ``total`` is the pose score.

    Raises:
        ValueError: If the ligand has no 3D conformer.
    """
    if params is None:
        params = EmpiricalParams()

    ligand_heavy = Chem.RemoveAllHs(ligand)
    if ligand_heavy.GetNumConformers() == 0:
        raise ValueError("ligand has no 3D conformer")
    ligand_coords = ligand_heavy.GetConformer().GetPositions()
    ligand_typing = atom_typing(ligand_heavy)

    # Surface distance d[i, j] = |lig_i - prot_j| - r_lig_i - r_prot_j.
    diff = ligand_coords[:, None, :] - protein_coords[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    d -= ligand_typing.radii[:, None] + protein_typing.radii[None, :]

    hydrophobic_pair = (
        ligand_typing.hydrophobic[:, None] & protein_typing.hydrophobic[None, :]
    )
    hbond_pair = (ligand_typing.donor[:, None] & protein_typing.acceptor[None, :]) | (
        ligand_typing.acceptor[:, None] & protein_typing.donor[None, :]
    )

    gauss1_raw, repulsion_raw, hydrophobic_raw, hbond_raw = _term_sums(
        d, hydrophobic_pair, hbond_pair, params
    )
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(ligand_heavy, strict=False)
    return ScoreComponents(
        gauss1_raw=gauss1_raw,
        repulsion_raw=repulsion_raw,
        hydrophobic_raw=hydrophobic_raw,
        hbond_raw=hbond_raw,
        params=params,
        n_rot=n_rot,
    )


# Rotatable single bond (not terminal, not triple-adjacent, not in a ring); amide
# and thioamide bonds are kept rigid. Both are cmxflow's get_rotatable_bonds SMARTS.
_ROTATABLE_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
_AMIDE_SMARTS = "[#7;X3]-[#6;X3]=[O,S]"


def _rigid_fragments(mol: Chem.Mol) -> np.ndarray:
    """Label each heavy atom by its rigid fragment.

    Cutting the molecule at its rotatable bonds splits it into rigid fragments:
    atoms in different fragments can move relative to each other, atoms in the
    same fragment cannot. Amide and thioamide bonds are kept rigid. This is
    cmxflow's rigid-fragment partition, used to select intramolecular pairs.

    Args:
        mol: Heavy-atom molecule.

    Returns:
        (N,) array of fragment labels (union-find roots), one per atom.
    """
    rotatable = Chem.MolFromSmarts(_ROTATABLE_SMARTS)
    amide = Chem.MolFromSmarts(_AMIDE_SMARTS)
    amide_bonds = {frozenset(m[:2]) for m in mol.GetSubstructMatches(amide)}
    rot_bonds = {
        frozenset((j, k))
        for j, k in mol.GetSubstructMatches(rotatable)
        if frozenset((j, k)) not in amide_bonds
    }

    parent = list(range(mol.GetNumAtoms()))

    def find(x: int) -> int:
        """The union-find root of atom ``x``, with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if frozenset((a, b)) in rot_bonds:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(mol.GetNumAtoms())])


def score_intramolecular(
    ligand: Chem.Mol, params: EmpiricalParams | None = None
) -> ScoreComponents:
    """Score a ligand's internal Vinardo energy over its non-bonded pairs.

    The same four Vinardo terms as the intermolecular score, summed over the
    ligand's own atom pairs that (1) are at least three bonds apart (1-4 and
    beyond) and (2) lie in different rigid fragments, so they move relative to
    each other under some torsion. This is cmxflow's (and Vina's) intramolecular
    pair selection. It is not a binding score: it measures internal strain and
    guides a pose settle, while the reported affinity stays intermolecular only.

    Args:
        ligand: The ligand molecule with a 3D conformer. Hydrogens are stripped.
        params: Scoring weights and cutoffs; the smina Vinardo defaults if None.

    Returns:
        The four term sums, whose ``total`` is the internal energy and whose
        ``repulsion`` is the internal clash strain.

    Raises:
        ValueError: If the ligand has no 3D conformer.
    """
    if params is None:
        params = EmpiricalParams()
    heavy = Chem.RemoveAllHs(ligand)
    if heavy.GetNumConformers() == 0:
        raise ValueError("ligand has no 3D conformer")
    coords = heavy.GetConformer().GetPositions()
    typing = atom_typing(heavy)
    valid = intramolecular_mask(heavy)

    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff * diff).sum(-1)) - typing.radii[:, None] - typing.radii[None, :]
    hydrophobic_pair = typing.hydrophobic[:, None] & typing.hydrophobic[None, :] & valid
    hbond_pair = (
        (typing.donor[:, None] & typing.acceptor[None, :])
        | (typing.acceptor[:, None] & typing.donor[None, :])
    ) & valid

    gauss1_raw, repulsion_raw, hydrophobic_raw, hbond_raw = _term_sums(
        d, hydrophobic_pair, hbond_pair, params, valid=valid
    )
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(heavy, strict=False)
    return ScoreComponents(
        gauss1_raw=gauss1_raw,
        repulsion_raw=repulsion_raw,
        hydrophobic_raw=hydrophobic_raw,
        hbond_raw=hbond_raw,
        params=params,
        n_rot=n_rot,
    )


def intramolecular_mask(mol: Chem.Mol) -> np.ndarray:
    """The ligand's scored intramolecular pair mask (upper-triangle boolean).

    A pair counts when it is at least three bonds apart (1-4 and beyond) and lies
    in different rigid fragments, so it moves relative under some torsion —
    cmxflow's (Vina's) selection. Depends only on the graph, so a torsion scan
    computes it once.

    Args:
        mol: Heavy-atom molecule.

    Returns:
        (N, N) boolean mask, upper triangle only.
    """
    n = mol.GetNumAtoms()
    fragment = _rigid_fragments(mol)
    bond_dist = Chem.GetDistanceMatrix(mol)
    return (
        np.triu(np.ones((n, n), dtype=bool), k=1)
        & (bond_dist >= 3)
        & (fragment[:, None] != fragment[None, :])
    )


def vinardo_objective(
    ligand_coords: np.ndarray,
    ligand_typing: AtomTyping,
    protein_coords: np.ndarray | None,
    protein_typing: AtomTyping | None,
    intra_mask: np.ndarray,
    n_rot: int,
    params: EmpiricalParams | None = None,
) -> float:
    """The Vina search energy: intermolecular + intramolecular Vinardo total.

    This is the pose-settle objective for ``minimize`` — not a binding score. The
    ligand typing, receptor data, intramolecular mask, and ``n_rot`` are fixed
    across a torsion scan, so a caller precomputes them once and passes new
    ``ligand_coords`` per turn.

    Args:
        ligand_coords: (N, 3) ligand heavy-atom coordinates for this turn.
        ligand_typing: Ligand Vinardo typing (aligned to ``ligand_coords``).
        protein_coords: (M, 3) receptor coordinates, or None for no receptor.
        protein_typing: Receptor typing, or None for no receptor.
        intra_mask: (N, N) intramolecular pair mask from ``intramolecular_mask``.
        n_rot: Ligand rotatable-bond count (the torsion divisor's input).
        params: Scoring weights and cutoffs; the smina Vinardo defaults if None.

    Returns:
        ``inter.total + intra.total`` (lower is better).
    """
    if params is None:
        params = EmpiricalParams()
    divisor = 1.0 + params.w_rot * n_rot

    total = 0.0
    if (
        protein_coords is not None
        and protein_typing is not None
        and len(protein_coords)
    ):
        diff = ligand_coords[:, None, :] - protein_coords[None, :, :]
        d = np.sqrt((diff * diff).sum(-1))
        d -= ligand_typing.radii[:, None] + protein_typing.radii[None, :]
        hydrophobic = (
            ligand_typing.hydrophobic[:, None] & protein_typing.hydrophobic[None, :]
        )
        hbond = (ligand_typing.donor[:, None] & protein_typing.acceptor[None, :]) | (
            ligand_typing.acceptor[:, None] & protein_typing.donor[None, :]
        )
        total += _weighted_total(
            _term_sums(d, hydrophobic, hbond, params), divisor, params
        )

    diff = ligand_coords[:, None, :] - ligand_coords[None, :, :]
    d = np.sqrt((diff * diff).sum(-1))
    d -= ligand_typing.radii[:, None] + ligand_typing.radii[None, :]
    hydrophobic = (
        ligand_typing.hydrophobic[:, None]
        & ligand_typing.hydrophobic[None, :]
        & intra_mask
    )
    hbond = (
        (ligand_typing.donor[:, None] & ligand_typing.acceptor[None, :])
        | (ligand_typing.acceptor[:, None] & ligand_typing.donor[None, :])
    ) & intra_mask
    total += _weighted_total(
        _term_sums(d, hydrophobic, hbond, params, valid=intra_mask), divisor, params
    )
    return total


def _weighted_total(
    raws: tuple[float, float, float, float], divisor: float, params: EmpiricalParams
) -> float:
    """Weight the four raw term sums and apply the torsion divisor."""
    gauss1, repulsion, hydrophobic, hbond = raws
    return (
        params.w_gauss1 * gauss1
        + params.w_repulsion * repulsion
        + params.w_hydrophobic * hydrophobic
        + params.w_hbond * hbond
    ) / divisor


def _term_sums(
    d: np.ndarray,
    hydrophobic_pair: np.ndarray,
    hbond_pair: np.ndarray,
    params: EmpiricalParams,
    valid: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Unweighted Vinardo term sums over atom pairs.

    Args:
        d: (N, M) surface distances between the two atom sets.
        hydrophobic_pair: (N, M) mask of hydrophobic pairs.
        hbond_pair: (N, M) mask of donor/acceptor pairs.
        params: The scoring cutoffs.
        valid: (N, M) mask of pairs to count at all. The gauss1 and repulsion
            terms apply to every pair by default, so a caller that scores a subset
            (the intramolecular pair list) passes this to gate them; the
            hydrophobic and hbond terms are already gated by their pair masks.
            None counts every pair (the intermolecular case).

    Returns:
        The raw ``(gauss1, repulsion, hydrophobic, hbond)`` sums.
    """
    z = (d - params.gauss1_offset) / params.gauss1_width
    gauss1 = np.exp(-(z**2))  # attraction over every pair

    repulsion = np.where(d < 0.0, d**2, 0.0)  # clash penalty over every pair

    if valid is not None:  # score only the selected pairs (intramolecular subset)
        gauss1 = np.where(valid, gauss1, 0.0)
        repulsion = np.where(valid, repulsion, 0.0)

    span = params.hydro_bad - params.hydro_good
    hydrophobic = np.where(
        d <= params.hydro_good,
        1.0,
        np.where(d < params.hydro_bad, (params.hydro_bad - d) / span, 0.0),
    )
    hydrophobic = np.where(hydrophobic_pair, hydrophobic, 0.0)

    hbond = np.where(
        d <= params.hbond_good,
        1.0,
        np.where(d < 0.0, -d / (-params.hbond_good), 0.0),
    )
    hbond = np.where(hbond_pair, hbond, 0.0)

    return (
        float(gauss1.sum()),
        float(repulsion.sum()),
        float(hydrophobic.sum()),
        float(hbond.sum()),
    )
