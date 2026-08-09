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
