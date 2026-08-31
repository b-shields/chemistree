"""Pure-numpy torsion geometry.

No RDKit, no tree: the rotation-about-axis matrix, the worst van der Waals overlap
between two atom sets, and assembling a torsion scan's per-degree scores into the
current, window-best, and global-best turns (ties resolved toward the requested
angle). The session layer supplies coordinates and radii, scores each turn, and
applies the chosen rotation.
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


def scan_result(
    scores: np.ndarray, *, target_deg: float, window_deg: float
) -> ScanResult:
    """Assemble a scan result from a score at each whole-degree turn.

    Lower scores are better; ties break toward ``target_deg``. Shared by the clash
    scan and the Vinardo scan, which differ only in how they score each turn.

    Args:
        scores: (360,) score at each whole-degree turn.
        target_deg: The requested turn; the window centres on it and ties resolve
            toward it.
        window_deg: Half-width of the window around ``target_deg``.

    Returns:
        The current, window-best, and global-best turns.
    """
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
