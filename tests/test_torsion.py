"""Pure geometry for torsion rotation: the axis matrix, clash score, and scan."""

import math

import numpy as np
import pytest

from chemistree.torsion import (
    axis_matrix,
    clash_score,
    scan_torsion,
    worst_overlap,
)


def _apply(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to a 3-vector."""
    return (matrix @ np.array([*point, 1.0]))[:3]


def test_axis_matrix_rotates_about_the_axis():
    # +90 deg about z takes the x-axis onto the y-axis (right-hand rule).
    matrix = axis_matrix(np.zeros(3), np.array([0.0, 0.0, 1.0]), math.radians(90))
    assert np.allclose(_apply(matrix, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0])


def test_axis_matrix_fixes_points_on_the_axis():
    # A point on the axis (through (0,0,5), along z) does not move.
    point = np.array([0.0, 0.0, 5.0])
    matrix = axis_matrix(point, np.array([0.0, 0.0, 1.0]), math.radians(137))
    assert np.allclose(_apply(matrix, point), point, atol=1e-9)


def test_axis_matrix_is_a_rigid_rotation():
    matrix = axis_matrix(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 0.0]), 0.7)
    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)


def test_clash_score_sums_vdw_overlap():
    # Two atoms (r=1.7) 2.0 A apart with a 0.4 A tolerance overlap by 1.0 A.
    moving = np.array([[0.0, 0.0, 0.0]])
    other = np.array([[2.0, 0.0, 0.0]])
    score = clash_score(moving, np.array([1.7]), other, np.array([1.7]), tol=0.4)
    assert score == 1.0


def test_clash_score_is_zero_when_clear():
    moving = np.array([[0.0, 0.0, 0.0]])
    other = np.array([[5.0, 0.0, 0.0]])
    assert clash_score(moving, np.array([1.7]), other, np.array([1.7])) == 0.0


def test_scan_torsion_finds_a_lower_clash_near_the_target():
    # A moving atom at (1,0,0) overlaps an atom at (1.2,0,0); rotating about z
    # swings it away. The clash falls monotonically toward 180 degrees.
    moving = np.array([[1.0, 0.0, 0.0]])
    other = np.array([[1.2, 0.0, 0.0]])
    result = scan_torsion(
        moving,
        axis_point=np.zeros(3),
        axis_dir=np.array([0.0, 0.0, 1.0]),
        moving_r=np.array([1.6]),
        other_xyz=other,
        other_r=np.array([1.6]),
        target_deg=170,
        window_deg=20,
        tol=0.0,
    )
    assert result.global_best[1] <= result.window_best[1]
    assert result.window_best[1] < result.current  # the search relieves the clash
    lo, hi = 150, 190
    assert lo <= result.window_best[0] <= hi  # stays near the requested angle


def test_worst_overlap_finds_the_tightest_pair():
    a_xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    b_xyz = np.array([[9.5, 0.0, 0.0]])  # overlaps the second a-atom by 0.5 (r=1.6)
    overlap, i, j = worst_overlap(
        a_xyz, np.array([1.6, 1.6]), b_xyz, np.array([1.6]), tol=0.0
    )
    assert (i, j) == (1, 0)
    assert overlap == pytest.approx(3.2 - 0.5)


def test_worst_overlap_is_empty_safe():
    assert worst_overlap(
        np.empty((0, 3)), np.empty(0), np.array([[0.0, 0.0, 0.0]]), np.array([1.6])
    ) == (0.0, -1, -1)


def test_scan_torsion_honors_the_target_when_nothing_clashes():
    # With no other atoms there is no clash at any angle, so the window best is the
    # requested angle itself (apply exactly what was asked).
    moving = np.array([[1.0, 0.0, 0.0]])
    result = scan_torsion(
        moving,
        axis_point=np.zeros(3),
        axis_dir=np.array([0.0, 0.0, 1.0]),
        moving_r=np.array([1.6]),
        other_xyz=np.empty((0, 3)),
        other_r=np.empty(0),
        target_deg=120,
        window_deg=60,
    )
    assert result.window_best == (120, 0.0)
