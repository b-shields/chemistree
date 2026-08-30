"""Geometric helpers for coordinate-level reasoning."""

from __future__ import annotations

import numpy as np
from rdkit import Chem


def chiral_volume(conf: Chem.Conformer, center: int, a: int, b: int, c: int) -> float:
    """Signed volume of the tetrahedron at ``center`` spanned by three neighbors.

    The sign is a label-independent measure of handedness: two configurations
    share chirality when their signed volumes (over corresponding neighbors) agree.

    Args:
        conf: Conformer holding the coordinates.
        center: Index of the central atom.
        a: Index of the first neighbor.
        b: Index of the second neighbor.
        c: Index of the third neighbor.

    Returns:
        The signed tetrahedron volume.
    """
    p = np.array(conf.GetAtomPosition(center))
    va = np.array(conf.GetAtomPosition(a)) - p
    vb = np.array(conf.GetAtomPosition(b)) - p
    vc = np.array(conf.GetAtomPosition(c)) - p
    return float(np.dot(va, np.cross(vb, vc)))


def rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """A 3x3 rotation matrix taking the direction of ``src`` onto that of ``dst``.

    Uses Rodrigues' formula about the axis perpendicular to both vectors. Parallel
    and antiparallel inputs are handled without dividing by zero.

    Args:
        src: Source direction vector (need not be unit length).
        dst: Target direction vector (need not be unit length).

    Returns:
        The rotation matrix ``R`` with ``R @ unit(src) == unit(dst)``.
    """
    a = src / np.linalg.norm(src)
    b = dst / np.linalg.norm(dst)
    axis = np.cross(a, b)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(a, b))
    if sine < 1e-8:
        if cosine > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 degrees about any axis perpendicular to ``a``.
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-8:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        flip: np.ndarray = 2.0 * np.outer(perp, perp) - np.eye(3)
        return flip
    axis /= sine
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation: np.ndarray = np.eye(3) + sine * k + (1.0 - cosine) * (k @ k)
    return rotation


def align_transform(
    from_point: np.ndarray,
    from_dir: np.ndarray,
    to_point: np.ndarray,
    to_dir: np.ndarray,
) -> np.ndarray:
    """A 4x4 rigid transform mapping a point and direction onto another.

    Rotates ``from_dir`` onto ``to_dir`` (the minimal rotation between them) and
    translates ``from_point`` onto ``to_point``. Suits
    ``rdMolTransforms.TransformConformer``, which left-multiplies a homogeneous
    ``[x, y, z, 1]`` column.

    Args:
        from_point: Point that maps onto ``to_point``.
        from_dir: Direction that rotates onto ``to_dir``.
        to_point: Where ``from_point`` lands.
        to_dir: Where ``from_dir`` points after the rotation.

    Returns:
        The 4x4 rigid transform matrix.
    """
    rotation = rotation_between(from_dir, to_dir)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = to_point - rotation @ from_point
    return matrix
