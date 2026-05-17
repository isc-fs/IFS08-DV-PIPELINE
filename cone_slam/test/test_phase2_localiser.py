"""Unit tests for Phase2Localiser (#496).

Pure-Python; no ROS, no GTSAM. Verifies the EKF math: predict
advances the pose deterministically and inflates covariance,
measurement updates pull the state toward map-consistent positions,
the Mahalanobis gate rejects outliers, and unmatched observations
leave the filter alone.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cone_slam.frozen_map import FrozenMap
from cone_slam.landmark_db import Landmark
from cone_slam.phase1_mapper import Observation, Pose2D
from cone_slam.phase2_localiser import Phase2Localiser, CHI2_95_2DOF


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _make_landmark(lid: int, x: float, y: float, *, n_obs: int = 5,
                   sigma: float = 0.10) -> Landmark:
    return Landmark(
        id=lid,
        position=np.array([x, y, 0.0]),
        n_observations=n_obs,
        last_seen_step=0,
        sigma_xy=sigma,
        is_big_orange=False,
    )


def _make_map(points: list[tuple[float, float]]) -> FrozenMap:
    landmarks = [_make_landmark(i, x, y) for i, (x, y) in enumerate(points)]
    return FrozenMap.from_landmarks(landmarks)


def _observe_from(pose: Pose2D, landmark_xy: tuple[float, float],
                  noise_xy: tuple[float, float] = (0.0, 0.0)) -> Observation:
    """Synthesise a body-frame observation of a world-frame cone
    from a given pose (with optional additive noise in body frame)."""
    lx, ly = landmark_xy
    dx, dy = lx - pose.x, ly - pose.y
    c, s = math.cos(-pose.yaw), math.sin(-pose.yaw)
    bx = c * dx - s * dy + noise_xy[0]
    by = s * dx + c * dy + noise_xy[1]
    return Observation(body_x=bx, body_y=by, sigma_m=0.20)


# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------

def test_predict_advances_pose_along_body_x_when_yaw_zero() -> None:
    """Yaw 0, drive 1 m forward in body x → world x advances by 1."""
    m = _make_map([(10.0, 0.0)])
    loc = Phase2Localiser(m, Pose2D(0.0, 0.0, 0.0))
    loc.predict(dx_body=1.0, dy_body=0.0, dtheta=0.0)
    assert loc.pose.x == pytest.approx(1.0)
    assert loc.pose.y == pytest.approx(0.0)
    assert loc.pose.yaw == pytest.approx(0.0)


def test_predict_rotates_body_motion_into_world() -> None:
    """Yaw 90°, drive 1 m forward in body → world y advances."""
    m = _make_map([(10.0, 0.0)])
    loc = Phase2Localiser(m, Pose2D(0.0, 0.0, math.pi / 2))
    loc.predict(dx_body=1.0, dy_body=0.0, dtheta=0.0)
    assert loc.pose.x == pytest.approx(0.0, abs=1e-9)
    assert loc.pose.y == pytest.approx(1.0)


def test_predict_inflates_covariance() -> None:
    """A predict tick must grow the position variance by at least Q."""
    m = _make_map([(10.0, 0.0)])
    loc = Phase2Localiser(m, Pose2D(0.0, 0.0, 0.0))
    p0 = loc.covariance
    loc.predict(dx_body=1.0, dy_body=0.0, dtheta=0.0,
                sigma_xy=0.10, sigma_yaw=math.radians(1.0))
    p1 = loc.covariance
    assert p1[0, 0] > p0[0, 0]
    assert p1[1, 1] > p0[1, 1]
    assert p1[2, 2] > p0[2, 2]


# ---------------------------------------------------------------------
# Update — happy path
# ---------------------------------------------------------------------

def test_update_with_perfect_observation_keeps_pose_and_shrinks_cov() -> None:
    """If the pose is correct and the observation is noise-free,
    the update should not move the pose (the innovation is zero)
    but should shrink the covariance — information went in."""
    landmark = (5.0, 0.0)
    m = _make_map([landmark])
    pose = Pose2D(0.0, 0.0, 0.0)
    loc = Phase2Localiser(m, pose)
    p0 = loc.covariance
    obs = _observe_from(pose, landmark)
    summary = loc.update([obs])

    assert summary.n_matched == 1
    assert summary.n_gated_out == 0
    assert loc.pose.x == pytest.approx(pose.x, abs=1e-6)
    assert loc.pose.y == pytest.approx(pose.y, abs=1e-6)
    assert loc.covariance[0, 0] < p0[0, 0]
    assert loc.covariance[1, 1] < p0[1, 1]


def test_update_corrects_lateral_drift() -> None:
    """Bias the initial pose 0.5 m to the side; observe two cones
    whose true positions are 5 m ahead at ±2 m lateral. The update
    should pull the pose back toward the map."""
    cones = [(5.0, 2.0), (5.0, -2.0)]
    m = _make_map(cones)
    true_pose = Pose2D(0.0, 0.0, 0.0)
    biased = Pose2D(0.0, 0.5, 0.0)         # 0.5 m off in +y
    loc = Phase2Localiser(m, biased)
    obs = [_observe_from(true_pose, c) for c in cones]
    summary = loc.update(obs)

    assert summary.n_matched == 2
    # The lateral error should shrink — not to exactly zero in a
    # single update (Kalman gain < 1) but meaningfully.
    assert abs(loc.pose.y) < 0.5
    assert abs(loc.pose.y) < abs(biased.y) - 1e-3


def test_update_converges_to_true_pose_after_many_scans() -> None:
    """Repeated noise-free observations of the same scene should
    pull a biased filter all the way to truth (modulo tiny float
    residuals)."""
    cones = [(5.0, 2.0), (5.0, -2.0), (3.0, 3.0), (7.0, -1.0)]
    m = _make_map(cones)
    true_pose = Pose2D(0.0, 0.0, 0.0)
    loc = Phase2Localiser(m, Pose2D(0.2, -0.3, math.radians(2.0)))
    for _ in range(30):
        obs = [_observe_from(true_pose, c) for c in cones]
        loc.update(obs)
    # Sub-centimetre / sub-degree convergence is the meaningful
    # signal; tighter tolerances would just be picking up
    # float-precision noise from repeated trig + matrix ops.
    assert loc.pose.x == pytest.approx(0.0, abs=1e-2)
    assert loc.pose.y == pytest.approx(0.0, abs=1e-2)
    assert loc.pose.yaw == pytest.approx(0.0, abs=math.radians(1.0))


# ---------------------------------------------------------------------
# DA / Mahalanobis gate
# ---------------------------------------------------------------------

def test_mahalanobis_rejects_wild_outlier() -> None:
    """An observation that lies dozens of σ away from any landmark
    must be gated out — it represents a false detection that
    shouldn't bias the state."""
    m = _make_map([(5.0, 0.0)])
    pose = Pose2D(0.0, 0.0, 0.0)
    loc = Phase2Localiser(
        m, pose,
        init_cov_diag=(0.05, 0.05, math.radians(1.0)),
        obs_sigma_m=0.20,
        max_match_radius_m=10.0,        # generous radius so the
                                        # outlier *gets* a candidate
                                        # and we test the gate, not
                                        # the radius cap.
    )
    # An "observation" that says the cone at (5,0) is actually
    # 2 m to the side. Under tight covariance this is a 10σ event.
    outlier = Observation(body_x=5.0, body_y=2.0)
    summary = loc.update([outlier])
    assert summary.n_matched == 0
    assert summary.n_gated_out == 1


def test_observation_beyond_radius_is_unmatched_not_gated() -> None:
    """The hard match-radius cap fires *before* the χ² test, so
    candidates that aren't found at all count as unmatched."""
    m = _make_map([(5.0, 0.0)])
    loc = Phase2Localiser(m, Pose2D(0.0, 0.0, 0.0),
                          max_match_radius_m=1.0)
    # An obs pointing at (5, 5) → no landmark within 1 m. Should
    # be unmatched, not gated.
    distant = Observation(body_x=5.0, body_y=5.0)
    summary = loc.update([distant])
    assert summary.n_unmatched == 1
    assert summary.n_gated_out == 0
    assert summary.n_matched == 0


def test_unmatched_observations_leave_state_alone() -> None:
    """Pose & covariance must not change when nothing matched."""
    m = _make_map([(5.0, 0.0)])
    loc = Phase2Localiser(m, Pose2D(0.0, 0.0, 0.0),
                          max_match_radius_m=1.0)
    x0 = loc.pose
    P0 = loc.covariance
    loc.update([Observation(body_x=5.0, body_y=5.0)])
    assert loc.pose.x == pytest.approx(x0.x)
    assert loc.pose.y == pytest.approx(x0.y)
    assert loc.pose.yaw == pytest.approx(x0.yaw)
    np.testing.assert_array_equal(loc.covariance, P0)


def test_multiple_matches_shrink_covariance_more_than_single() -> None:
    """Information-content sanity: two gated matches should reduce
    position variance more than one match would."""
    cones = [(5.0, 2.0), (5.0, -2.0)]
    m = _make_map(cones)
    pose = Pose2D(0.0, 0.0, 0.0)

    loc_one = Phase2Localiser(m, pose)
    loc_one.update([_observe_from(pose, cones[0])])

    loc_two = Phase2Localiser(m, pose)
    loc_two.update([_observe_from(pose, c) for c in cones])

    trace_one = float(np.trace(loc_one.covariance[:2, :2]))
    trace_two = float(np.trace(loc_two.covariance[:2, :2]))
    assert trace_two < trace_one


# ---------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------

def test_empty_map_is_no_op() -> None:
    """An empty FrozenMap should not crash and should produce zero
    matches."""
    empty = FrozenMap.from_landmarks([])
    loc = Phase2Localiser(empty, Pose2D(0.0, 0.0, 0.0))
    summary = loc.update([Observation(body_x=5.0, body_y=0.0)])
    assert summary.n_matched == 0
    assert summary.n_unmatched == 0  # short-circuit: skipped before DA
    assert summary.n_obs == 1


def test_empty_observations_returns_zero_summary() -> None:
    m = _make_map([(5.0, 0.0)])
    loc = Phase2Localiser(m, Pose2D(0.0, 0.0, 0.0))
    summary = loc.update([])
    assert summary.n_obs == 0
    assert summary.n_matched == 0


# ---------------------------------------------------------------------
# Predict + update combined — the full filter loop
# ---------------------------------------------------------------------

def test_predict_then_update_recovers_drift() -> None:
    """Simulate odometry drift: predict 10 m forward with body-x
    motion but the true motion is also accumulating lateral error.
    Cone observations of the static map should pull the pose back
    when the update runs."""
    cones = [(15.0, 2.0), (15.0, -2.0), (5.0, 2.0), (5.0, -2.0)]
    m = _make_map(cones)

    loc = Phase2Localiser(m, Pose2D(0.0, 0.0, 0.0),
                          init_cov_diag=(0.05, 0.05, math.radians(1.0)))
    # Simulate 10 small predict ticks that accumulate +0.4 m of
    # lateral drift in the world (the filter doesn't know).
    drifted_true = Pose2D(0.0, 0.0, 0.0)
    for _ in range(10):
        loc.predict(dx_body=1.0, dy_body=0.04, dtheta=0.0,
                    sigma_xy=0.05, sigma_yaw=math.radians(0.5))
        # True motion: 1 m forward, but no lateral drift in truth.
        drifted_true = Pose2D(drifted_true.x + 1.0,
                              drifted_true.y, drifted_true.yaw)
    # The filter now thinks it's at y ≈ 0.4 m, truth is y = 0.
    assert loc.pose.y > 0.3
    obs = [_observe_from(drifted_true, c) for c in cones]
    loc.update(obs)
    # After one update with four matches the filter should have
    # corrected most of that lateral error.
    assert abs(loc.pose.y) < 0.1
    assert loc.pose.x == pytest.approx(drifted_true.x, abs=0.2)
