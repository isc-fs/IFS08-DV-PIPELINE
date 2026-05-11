"""Tests for the new-landmark proximity veto.

The cone-graph cascade signature observed on bag
`lap_attempt_20260511_142317` at t≈144.41 s was: pose drift creeps to
~1 m (≈ DISTANCE_GATE_M); ~half the observations fall just past the
DA gate (1.02-1.20 m to their nearest landmark); they get spawned as
ghost landmarks at the drifted pose; the factor graph diverges.

These tests pin the `LandmarkDb.nearest_xy_distance_m` contract that
the SLAM node uses to veto those ghost spawns.
"""
from __future__ import annotations

import numpy as np

from cone_slam.landmark_db import LandmarkDb


def test_empty_db_returns_infinity() -> None:
    """No landmarks → no nearest → +inf (never vetoes anything)."""
    db = LandmarkDb()
    d = db.nearest_xy_distance_m(np.array([1.0, 2.0, 0.0]))
    assert d == float("inf")


def test_single_landmark_distance() -> None:
    db = LandmarkDb()
    db.create(np.array([3.0, 4.0, 0.0]), step=0)
    d = db.nearest_xy_distance_m(np.array([0.0, 0.0, 0.0]))
    assert d == 5.0  # 3-4-5 triangle


def test_z_is_ignored() -> None:
    """Cone landmarks are XY-only for FSD tracks — Z mismatch must
    not inflate the proximity distance."""
    db = LandmarkDb()
    db.create(np.array([0.0, 0.0, 0.0]), step=0)
    d = db.nearest_xy_distance_m(np.array([0.0, 0.0, 100.0]))
    assert d == 0.0


def test_returns_closest_when_many_landmarks() -> None:
    db = LandmarkDb()
    for x in (10.0, 20.0, 30.0):
        db.create(np.array([x, 0.0, 0.0]), step=0)
    db.create(np.array([5.0, 0.5, 0.0]), step=0)  # closest
    d = db.nearest_xy_distance_m(np.array([5.0, 0.0, 0.0]))
    assert d == 0.5


def test_cascade_signature_vetoed_at_1p5m() -> None:
    """Reproduce the t=144.41 burst: 5 of 6 "new" observations had
    nearest-existing distances in [1.02, 1.20] m. A veto threshold of
    1.5 m must reject all 5 of those; the 6th (d=2.18 m, a plausible
    real new cone) must pass."""
    db = LandmarkDb()
    db.create(np.array([0.0, 0.0, 0.0]), step=0)

    # Simulated would-be-new world positions from the bag (relative
    # distances reproduced; absolute coordinates don't matter for
    # nearest-distance semantics).
    ghost_dists = [1.028, 1.066, 1.076, 1.155, 1.203]
    real_new_dist = 2.179
    veto_m = 1.5

    for d in ghost_dists:
        # Place candidate at d metres from landmark 0.
        xy = np.array([d, 0.0, 0.0])
        assert db.nearest_xy_distance_m(xy) == d
        assert db.nearest_xy_distance_m(xy) < veto_m, (
            f"cascade ghost at d={d:.3f} m must be vetoed by "
            f"threshold {veto_m}")

    # The plausible real-new cone at >2 m must pass through.
    xy_real = np.array([real_new_dist, 0.0, 0.0])
    assert db.nearest_xy_distance_m(xy_real) >= veto_m, (
        "a 2.18 m-distant candidate must NOT be vetoed at 1.5 m")
