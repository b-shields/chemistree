"""Geometry for torsion rotation and clash scoring.

Pure numpy: no RDKit, no tree. A torsion rotates a rigid set of atoms about an
axis; a clash score sums the van der Waals overlap between those atoms and their
surroundings; a scan evaluates every one-degree turn to find the least-clashing
angle near a requested one. The session layer supplies the coordinates and radii
and applies the chosen rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def axis_matrix(point: np.ndarray, direction: np.ndarray, radians: float) -> np.ndarray:
    """A 4x4 transform that rotates about an axis through a point.

    The matrix suits ``rdMolTransforms.TransformConformer``: it left-multiplies a
    homogeneous ``[x, y, z, 1]`` column.

    Args:
        point: A point the rotation axis passes through.
        direction: The axis direction (need not be unit length).
        radians: Rotation angle, right-handed about ``direction``.

    Returns:
        The 4x4 rotation-about-axis matrix.
    """
    x, y, z = direction / np.linalg.norm(direction)
    c, s = math.cos(radians), math.sin(radians)
    d = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * d, x * y * d - z * s, x * z * d + y * s],
            [y * x * d + z * s, c + y * y * d, y * z * d - x * s],
            [z * x * d - y * s, z * y * d + x * s, c + z * z * d],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = point - rotation @ point  # rotate about ``point``, not the origin
    return matrix


def clash_score(
    moving_xyz: np.ndarray,
    moving_r: np.ndarray,
    other_xyz: np.ndarray,
    other_r: np.ndarray,
    *,
    tol: float = 0.4,
) -> float:
    """Total van der Waals overlap between two sets of atoms.

    Each pair contributes ``max(0, (r_i + r_j) - tol - d)``: how far inside their
    combined van der Waals radii the atoms sit, past a tolerance. A clear structure
    scores 0; overlaps add up.

    Args:
        moving_xyz: (M, 3) coordinates of the moving atoms.
        moving_r: (M,) van der Waals radii of the moving atoms.
        other_xyz: (O, 3) coordinates of the surrounding atoms.
        other_r: (O,) van der Waals radii of the surrounding atoms.
        tol: Overlap allowed before a pair counts, in angstrom.

    Returns:
        The summed overlap in angstrom.
    """
    if len(moving_xyz) == 0 or len(other_xyz) == 0:
        return 0.0
    diff = moving_xyz[:, None, :] - other_xyz[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    contact = moving_r[:, None] + other_r[None, :] - tol
    return float(np.maximum(contact - dist, 0.0).sum())


def worst_overlap(
    a_xyz: np.ndarray,
    a_r: np.ndarray,
    b_xyz: np.ndarray,
    b_r: np.ndarray,
    *,
    tol: float = 0.4,
) -> tuple[float, int, int]:
    """The single most-overlapping atom pair between two sets.

    Args:
        a_xyz: (A, 3) coordinates of the first set.
        a_r: (A,) van der Waals radii of the first set.
        b_xyz: (B, 3) coordinates of the second set.
        b_r: (B,) van der Waals radii of the second set.
        tol: Overlap allowed before a pair counts, in angstrom.

    Returns:
        ``(overlap, i, j)``: the largest van der Waals overlap and the indices of
        the atoms in each set that make it. ``(0.0, -1, -1)`` when either set is
        empty.
    """
    if len(a_xyz) == 0 or len(b_xyz) == 0:
        return 0.0, -1, -1
    diff = a_xyz[:, None, :] - b_xyz[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    overlap = a_r[:, None] + b_r[None, :] - tol - dist
    i, j = np.unravel_index(int(np.argmax(overlap)), overlap.shape)
    return float(overlap[i, j]), int(i), int(j)


@dataclass(frozen=True)
class ScanResult:
    """The outcome of a one-degree torsion scan.

    Attributes:
        current: Clash score at the current pose (a zero-degree turn).
        window_best: ``(degrees, score)`` of the least-clashing turn within the
            window around the requested angle. This is the turn to apply.
        global_best: ``(degrees, score)`` of the least-clashing turn over the full
            circle, reported so a larger move can be suggested.
    """

    current: float
    window_best: tuple[int, float]
    global_best: tuple[int, float]


def scan_torsion(
    moving_xyz: np.ndarray,
    *,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    moving_r: np.ndarray,
    other_xyz: np.ndarray,
    other_r: np.ndarray,
    target_deg: float,
    window_deg: float,
    tol: float = 0.4,
) -> ScanResult:
    """Score every one-degree turn and pick the best near the target and overall.

    The moving atoms are rotated about the axis by each whole-degree offset from
    the current pose; each turn is scored against the surrounding atoms. Ties
    resolve toward the requested angle, so a clear structure applies exactly what
    was asked.

    Args:
        moving_xyz: (M, 3) coordinates of the atoms that turn.
        axis_point: A point the rotation axis passes through.
        axis_dir: The axis direction.
        moving_r: (M,) van der Waals radii of the moving atoms.
        other_xyz: (O, 3) coordinates of the surrounding atoms.
        other_r: (O,) van der Waals radii of the surrounding atoms.
        target_deg: The requested turn, in degrees.
        window_deg: Half-width of the search window around ``target_deg``.
        tol: Overlap tolerance passed to the clash score.

    Returns:
        The scan result: current, window-best, and global-best turns.
    """
    scores = _scan_scores(
        moving_xyz, axis_point, axis_dir, moving_r, other_xyz, other_r, tol
    )
    window = [
        deg % 360
        for deg in range(
            round(target_deg - window_deg), round(target_deg + window_deg) + 1
        )
    ]
    return ScanResult(
        current=float(scores[0]),
        window_best=_best(scores, window, target_deg),
        global_best=_best(scores, list(range(360)), target_deg),
    )


def _scan_scores(
    moving_xyz: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    moving_r: np.ndarray,
    other_xyz: np.ndarray,
    other_r: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Clash score at each of the 360 one-degree turns, computed in one pass.

    Only atoms a turn could bring into contact are scored: as the moving atoms
    sweep circles about the axis, a surrounding atom can overlap one only if it
    lies within their combined radii of that circle. Atoms outside every circle's
    reach are dropped before the scan.
    """
    if len(moving_xyz) == 0 or len(other_xyz) == 0:
        return np.zeros(360)
    unit = axis_dir / np.linalg.norm(axis_dir)
    m_local = moving_xyz - axis_point
    m_axial = m_local @ unit  # position along the axis (fixed under the turn)
    o_local = other_xyz - axis_point
    o_axial = o_local @ unit
    m_radial = np.linalg.norm(m_local - np.outer(m_axial, unit), axis=1)
    o_radial = np.linalg.norm(o_local - np.outer(o_axial, unit), axis=1)

    # Closest a surrounding atom can get to a moving atom's swept circle.
    axial_gap = m_axial[:, None] - o_axial[None, :]
    radial_gap = m_radial[:, None] - o_radial[None, :]
    reach = np.sqrt(axial_gap**2 + radial_gap**2)
    contact = moving_r[:, None] + other_r[None, :] - tol
    keep = (reach < contact).any(axis=0)
    if not keep.any():
        return np.zeros(360)
    other_xyz, other_r = other_xyz[keep], other_r[keep]

    # Rotate every moving atom by every whole degree (Rodrigues, batched).
    theta = np.radians(np.arange(360))
    cos, sin = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    perp_cross = np.cross(unit, m_local)
    rotated = (
        m_local[None] * cos
        + perp_cross[None] * sin
        + unit[None, None, :] * (m_axial[None, :, None] * (1.0 - cos))
        + axis_point
    )  # (360, M, 3)

    # Pairwise distances via |a-b|^2 = |a|^2 + |b|^2 - 2 a.b, avoiding a huge tensor.
    a2 = (rotated**2).sum(-1)
    b2 = (other_xyz**2).sum(-1)
    dist = np.sqrt(
        np.maximum(
            a2[:, :, None] + b2[None, None, :] - 2 * (rotated @ other_xyz.T), 0.0
        )
    )
    pair_contact = moving_r[:, None] + other_r[None, :] - tol
    return np.maximum(pair_contact[None] - dist, 0.0).sum(axis=(1, 2))


def _best(
    scores: np.ndarray, candidates: list[int], target_deg: float
) -> tuple[int, float]:
    """The candidate angle with the least clash, ties broken toward the target."""
    best = min(
        candidates, key=lambda deg: (scores[deg], _circular_gap(deg, target_deg))
    )
    return best, float(scores[best])


def _circular_gap(deg: int, target_deg: float) -> float:
    """Smallest absolute angular distance from ``deg`` to ``target_deg``, in degrees."""
    gap = abs(deg - target_deg) % 360
    return min(gap, 360 - gap)
