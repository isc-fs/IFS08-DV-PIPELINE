"""Unit tests for Phase1Mapper (#496).

Pure-Python tests — no ROS, no GTSAM. Validates the core mapping
algorithm: project body-frame observations into world frame using
the supplied pose, associate to existing landmarks within the DA
gate, spawn new landmarks otherwise.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cone_slam.landmark_db import LandmarkDb
from cone_slam.phase1_mapper import Observation, Phase1Mapper, Pose2D


def _make_mapper(**kwargs) -> Phase1Mapper:
    db = LandmarkDb()
    return Phase1Mapper(db, **kwargs)


# ---------------------------------------------------------------------
# Projection: body → world
# ---------------------------------------------------------------------

def test_observation_at_origin_projects_directly() -> None:
    """Pose at origin, yaw 0 → world position == body position."""
    m = _make_mapper()
    m.observe_scan(
        Pose2D(0.0, 0.0, 0.0),
        [Observation(body_x=3.0, body_y=1.0)],
    )
    assert len(m.db) == 1
    lm = list(m.db)[0]
    assert lm.position[0] == pytest.approx(3.0)
    assert lm.position[1] == pytest.approx(1.0)


def test_yaw_90_rotates_body_to_world_correctly() -> None:
    """Car facing +y at world origin → a body-frame +x observation
    lands at world +y."""
    m = _make_mapper()
    m.observe_scan(
        Pose2D(0.0, 0.0, math.pi / 2),
        [Observation(body_x=2.0, body_y=0.0)],
    )
    lm = list(m.db)[0]
    assert lm.position[0] == pytest.approx(0.0, abs=1e-9)
    assert lm.position[1] == pytest.approx(2.0)


def test_pose_translation_offsets_landmark() -> None:
    """Car at (10, 20) yaw 0 → observation at body (1, 2) lands at
    world (11, 22)."""
    m = _make_mapper()
    m.observe_scan(
        Pose2D(10.0, 20.0, 0.0),
        [Observation(body_x=1.0, body_y=2.0)],
    )
    lm = list(m.db)[0]
    assert lm.position[0] == pytest.approx(11.0)
    assert lm.position[1] == pytest.approx(22.0)


# ---------------------------------------------------------------------
# Association vs new-landmark decision
# ---------------------------------------------------------------------

def test_second_observation_of_same_cone_associates() -> None:
    """Same world-frame cone observed twice → one landmark, not two."""
    m = _make_mapper(da_gate_m=1.0)
    pose = Pose2D(0.0, 0.0, 0.0)
    obs1 = Observation(body_x=5.0, body_y=0.0, sigma_m=0.20)
    obs2 = Observation(body_x=5.05, body_y=0.05, sigma_m=0.20)
    s1 = m.observe_scan(pose, [obs1])
    s2 = m.observe_scan(pose, [obs2])
    assert s1["n_new"] == 1 and s1["n_assoc"] == 0
    assert s2["n_new"] == 0 and s2["n_assoc"] == 1
    assert len(m.db) == 1


def test_far_apart_observations_spawn_separate_landmarks() -> None:
    """Two cones outside the DA gate → two separate landmarks."""
    m = _make_mapper(da_gate_m=1.0)
    pose = Pose2D(0.0, 0.0, 0.0)
    obs = [
        Observation(body_x=5.0, body_y=0.0),
        Observation(body_x=5.0, body_y=3.0),
    ]
    s = m.observe_scan(pose, obs)
    assert s["n_new"] == 2 and s["n_assoc"] == 0
    assert len(m.db) == 2


def test_da_gate_boundary_is_inclusive() -> None:
    """An observation exactly at the gate radius should associate."""
    m = _make_mapper(da_gate_m=1.0)
    pose = Pose2D(0.0, 0.0, 0.0)
    m.observe_scan(pose, [Observation(body_x=5.0, body_y=0.0)])
    s = m.observe_scan(pose, [Observation(body_x=6.0, body_y=0.0)])
    # Distance == gate; should associate, not spawn.
    assert s["n_new"] == 0 and s["n_assoc"] == 1


# ---------------------------------------------------------------------
# σ-weighted running mean — position converges, sigma drops
# ---------------------------------------------------------------------

def test_repeated_observations_shrink_sigma() -> None:
    """After N equal-sigma observations of a landmark, sigma should
    drop like σ₀ / √N."""
    m = _make_mapper()
    pose = Pose2D(0.0, 0.0, 0.0)
    sigma0 = 0.20
    n = 9   # √9 = 3 → expected sigma ≈ σ₀ / 3 = 0.067
    for _ in range(n):
        m.observe_scan(pose, [Observation(body_x=5.0, body_y=0.0, sigma_m=sigma0)])
    lm = list(m.db)[0]
    # The first observation seeds the landmark with sigma_xy = σ₀.
    # Then 8 more updates apply the inverse-variance mean. Expected:
    # 1/σ_final² = N / σ₀² → σ_final = σ₀ / √N
    expected = sigma0 / math.sqrt(n)
    assert lm.sigma_xy == pytest.approx(expected, rel=0.05)


def test_running_mean_converges_to_observed_position() -> None:
    """The σ-weighted mean of equally-weighted observations equals
    their plain mean."""
    m = _make_mapper()
    pose = Pose2D(0.0, 0.0, 0.0)
    body_xs = [5.0, 5.4, 4.8, 5.2, 5.1]
    for bx in body_xs:
        m.observe_scan(pose, [Observation(body_x=bx, body_y=0.0, sigma_m=0.20)])
    lm = list(m.db)[0]
    assert lm.position[0] == pytest.approx(np.mean(body_xs), rel=0.02)


# ---------------------------------------------------------------------
# Big-orange tagging
# ---------------------------------------------------------------------

def test_big_orange_flag_survives_association() -> None:
    """An observation tagged big_orange on the SECOND visit should
    flip the landmark's flag, not the first observation alone."""
    m = _make_mapper()
    pose = Pose2D(0.0, 0.0, 0.0)
    m.observe_scan(pose, [Observation(body_x=5.0, body_y=0.0,
                                       is_big_orange=False)])
    m.observe_scan(pose, [Observation(body_x=5.0, body_y=0.0,
                                       is_big_orange=True)])
    lm = list(m.db)[0]
    assert lm.is_big_orange


def test_big_orange_flag_wins_on_true() -> None:
    """Once tagged big_orange, a later False observation must not
    unset the flag — that would cause the lap detector to lose the
    gate mid-run if perception briefly mis-classifies."""
    m = _make_mapper()
    pose = Pose2D(0.0, 0.0, 0.0)
    m.observe_scan(pose, [Observation(body_x=5.0, body_y=0.0, is_big_orange=True)])
    m.observe_scan(pose, [Observation(body_x=5.0, body_y=0.0, is_big_orange=False)])
    lm = list(m.db)[0]
    assert lm.is_big_orange


# ---------------------------------------------------------------------
# Step counter + summary shape
# ---------------------------------------------------------------------

def test_step_increments_per_scan() -> None:
    """Per-scan step counter is what `last_seen_step` reads."""
    m = _make_mapper()
    pose = Pose2D(0.0, 0.0, 0.0)
    for _ in range(7):
        m.observe_scan(pose, [Observation(body_x=5.0, body_y=0.0)])
    assert m.step == 7
    lm = list(m.db)[0]
    assert lm.last_seen_step == 6  # 0-indexed; last update at step 6


def test_observe_scan_summary_keys() -> None:
    """Sanity: caller-facing summary dict exposes the same fields
    cone_graph_slam's SLAM_OBS log expected."""
    m = _make_mapper()
    s = m.observe_scan(Pose2D(0.0, 0.0, 0.0),
                       [Observation(body_x=3.0, body_y=1.0)])
    for key in ("step", "n_obs", "n_assoc", "n_new", "n_map", "n_big_orange"):
        assert key in s


# ---------------------------------------------------------------------
# Snapshot for Phase 2 handoff
# ---------------------------------------------------------------------

def test_snapshot_returns_landmark_list() -> None:
    m = _make_mapper()
    pose = Pose2D(0.0, 0.0, 0.0)
    m.observe_scan(pose, [Observation(body_x=3.0, body_y=0.0)])
    m.observe_scan(pose, [Observation(body_x=3.0, body_y=2.0)])  # new
    snap = m.snapshot_for_freeze()
    assert len(snap) == 2
    # Snapshot is a shallow list — Landmark refs into the DB, not
    # copies. Verify by attribute identity.
    db_ids = sorted(lm.id for lm in m.db)
    snap_ids = sorted(lm.id for lm in snap)
    assert db_ids == snap_ids
