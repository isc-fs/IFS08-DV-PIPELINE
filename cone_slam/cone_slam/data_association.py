"""Data association: match this scan's cone observations to existing
landmarks (or flag as new).

Position-only DA. The colour gate is gone, and so is the entire
upstream classifier (`color_classifier.classify`). Background:

  - The classifier was a body_y-sign + height heuristic with no real
    colour signal behind it (UE5 sets `bReturnPhysicalMaterial=false`
    at the LiDAR raycast site, the wire format is XYZ-only — see the
    audit summarised in PR #268).
  - With colour-blind FaSTTUBe sort (PR #268), the path planner doesn't
    rely on the classifier either.
  - The cross-colour DA gate (#272) was a band-aid for body_y flicker;
    with the classifier gone, all matches are equivalent.

Algorithm:
  1. Project every landmark into body frame.
  2. Build one (n_obs × n_lm) cost matrix:
       - Per-pair Euclidean offset.
       - Reject pairs above DISTANCE_GATE_M.
       - Reject pairs that fail the Mahalanobis χ² gate.
       - Surviving pairs get a finite Euclidean cost.
  3. One Hungarian assignment over the whole cost matrix.
  4. Matches with finite cost get a landmark_id; the rest get -1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from cone_slam.landmark_db import LandmarkDb


# Single Euclidean gate. Tight enough that adjacent cones in an FS
# corridor (≥ 3 m apart) don't cross-match even when pose has drifted,
# loose enough that legitimate same-cone re-observations under typical
# 100 ms-scan pose drift land within. The cross-colour 1.0 m value
# from #272 worked well in practice; with colour gone, the same
# threshold applies uniformly.
DISTANCE_GATE_M = 1.0


# Time-since-association gate expansion. Disabled (cap = 1.0×) — kept
# wired through `current_step` for easy re-enabling. See the prior
# experiment notes in git history if you're considering turning it on.
GATE_EXPANSION_PER_SCAN = 0.0
GATE_EXPANSION_CAP      = 1.0


# χ² threshold for the Mahalanobis gate. 2 DOF (xy in body frame).
#   95 % → 5.99
#   99 % → 9.21  (default, generous for early-iteration sparse data)
#  99.9 % → 13.82
MAHALANOBIS_CHI2 = 9.21


# Default per-landmark covariance to use when iSAM2 hasn't returned a
# marginal yet (brand-new landmark, same scan it was created). 0.5 m
# 1σ in each axis ≈ a generous "don't really know yet"; combined with
# the observation σ it's effectively the Euclidean fallback.
DEFAULT_LANDMARK_SIGMA_M = 0.5

# Default observation σ when the detector didn't report one
# (sigma_xy ≤ 0). 0.20 m matches the constant the SLAM graph used
# before per-cone σ propagation was wired up.
DEFAULT_OBS_SIGMA_M = 0.20


# === Covariance inflation for Mahalanobis gating ============================
# iSAM2's marginal covariance is the optimizer's INTERNAL certainty,
# not the true error. With a good IMU + cones, iSAM2 reports σ ~ 1 cm
# even when the actual pose has drifted 1+ m due to model errors,
# unmodeled bias drift, etc. Two prior attempts to enable Mahalanobis
# DA without inflation cascaded immediately because the gate
# collapsed to ~0.3 m on tightly-constrained landmarks. The fix is to
# (a) multiply Σ by a conservative factor before using it for gating,
# and (b) impose a per-component variance floor so the gate never
# collapses below physical cone-spacing bounds.
POSE_COV_INFLATION    = 16.0  # multiplier on iSAM2's pose marginal
LANDMARK_COV_INFLATION = 16.0  # multiplier on iSAM2's landmark marginal
COV_FLOOR_VAR_M2      = 0.49  # (0.7 m)² floor on diagonal of Σ_innov


@dataclass
class Observation:
    """One cone observation in body frame.

    `sigma_xy` is the detector-reported position uncertainty in metres
    (centroid SE scaled by sqrt(N) and range). A non-positive value is
    a sentinel meaning "detector didn't report σ — fall back to the
    SLAM-side range-only formula".
    """

    body_x: float
    body_y: float
    height: float
    sigma_xy: float = -1.0


@dataclass
class Match:
    """Result of the assignment step.

    `landmark_id == -1` means "new landmark" — caller should allocate
    one in LandmarkDb and add a factor to it instead of an existing one.
    """

    obs_index: int
    landmark_id: int


def _world_to_body(
    pose_x: float, pose_y: float, pose_yaw: float,
    world_xy: np.ndarray,
) -> np.ndarray:
    """Project a world-frame point into body frame given the car's 2D
    pose. Returns (2,) array of body-frame x, y."""
    dx = world_xy[0] - pose_x
    dy = world_xy[1] - pose_y
    c = np.cos(pose_yaw)
    s = np.sin(pose_yaw)
    return np.array([dx * c + dy * s, -dx * s + dy * c])


def associate(
    observations: List[Observation],
    pose_x: float,
    pose_y: float,
    pose_yaw: float,
    db: LandmarkDb,
    landmark_covariance_fn: Optional[
        Callable[[int], Optional[np.ndarray]]] = None,
    pose_xy_yaw_cov: Optional[np.ndarray] = None,
    current_step: int = -1,
) -> List[Match]:
    """Match observations to landmarks. One Match per observation.

    See module docstring for algorithm.

    Args:
        observations: list of cone observations in body frame.
        pose_x, pose_y, pose_yaw: current pose estimate (2D, world frame).
        db: landmark database.
        landmark_covariance_fn: optional callback for the marginal
            covariance of L(id) (typically FactorGraph.landmark_covariance).
        pose_xy_yaw_cov: optional 3×3 marginal of the predicted pose
            for Mahalanobis Jacobian propagation. None → ignored.
        current_step: optional, plumbed through for staleness gating.

    Returns:
        List of Match, in the same order as observations. landmark_id==-1
        means "no existing landmark — allocate new".
    """
    matches: List[Match] = [Match(obs_index=i, landmark_id=-1)
                            for i in range(len(observations))]

    if not observations:
        return matches

    candidates = list(db)
    if not candidates:
        return matches

    # World→body rotation: if R_b2w = [[c, -s], [s, c]] (yaw), then
    # R_w2b = R_b2w^T = [[c, s], [-s, c]].
    c_yaw = float(np.cos(pose_yaw))
    s_yaw = float(np.sin(pose_yaw))
    R_w2b = np.array([[ c_yaw, s_yaw],
                      [-s_yaw, c_yaw]])

    default_lm_cov_body = (DEFAULT_LANDMARK_SIGMA_M ** 2) * np.eye(2)
    default_obs_var = DEFAULT_OBS_SIGMA_M ** 2

    n_obs = len(observations)
    n_lm = len(candidates)
    cost = np.full((n_obs, n_lm), np.inf)

    # Pre-fetch landmark body-frame positions and (optionally) body-frame
    # covariances. The covariance call is potentially expensive (iSAM2
    # marginal computation) — once per landmark, not per (obs × lm).
    lm_bodies = np.empty((n_lm, 2))
    lm_cov_body = [default_lm_cov_body] * n_lm
    for j, lm in enumerate(candidates):
        lm_bodies[j] = _world_to_body(pose_x, pose_y, pose_yaw,
                                      lm.position[:2])
        if landmark_covariance_fn is not None:
            cov_world = landmark_covariance_fn(lm.id)
            if cov_world is not None:
                sigma_world_xy = LANDMARK_COV_INFLATION * cov_world[:2, :2]
                lm_cov_body[j] = R_w2b @ sigma_world_xy @ R_w2b.T

    for j in range(n_lm):
        lm = candidates[j]
        lm_body = lm_bodies[j]
        sigma_lm = lm_cov_body[j]
        # Per-landmark gate expansion based on staleness (disabled via
        # GATE_EXPANSION_PER_SCAN=0 — see config block above).
        if current_step >= 0:
            stale = max(0, current_step - lm.last_seen_step)
            gate_mult = min(
                GATE_EXPANSION_CAP,
                1.0 + GATE_EXPANSION_PER_SCAN * stale)
        else:
            gate_mult = 1.0
        gate_eff = DISTANCE_GATE_M * gate_mult

        # Pose-uncertainty contribution to Σ_innov via the Jacobian of
        # body-frame projection w.r.t. (yaw, x_w, y_w).
        if pose_xy_yaw_cov is not None:
            J = np.array([
                [ lm_body[1], -c_yaw, -s_yaw],
                [-lm_body[0],  s_yaw, -c_yaw],
            ])
            sigma_pose_contrib = (
                POSE_COV_INFLATION * (J @ pose_xy_yaw_cov @ J.T))
        else:
            sigma_pose_contrib = np.zeros((2, 2))

        for i in range(n_obs):
            o = observations[i]
            dx = o.body_x - lm_body[0]
            dy = o.body_y - lm_body[1]
            d_eu = float(np.hypot(dx, dy))
            if d_eu > gate_eff:
                continue

            obs_var = (o.sigma_xy ** 2) if o.sigma_xy > 0 \
                else default_obs_var
            sigma_innov = (sigma_lm
                           + obs_var * np.eye(2)
                           + sigma_pose_contrib
                           + COV_FLOOR_VAR_M2 * np.eye(2))
            innov = np.array([dx, dy])
            try:
                sol = np.linalg.solve(sigma_innov, innov)
            except np.linalg.LinAlgError:
                continue
            d2 = float(innov @ sol)
            if d2 <= MAHALANOBIS_CHI2:
                # Mahalanobis as gate, Euclidean as Hungarian cost.
                cost[i, j] = d_eu

    # Hungarian doesn't accept +∞; replace with a large finite value.
    big = 1e6
    cost_solver = np.where(np.isinf(cost), big, cost)

    row_ind, col_ind = linear_sum_assignment(cost_solver)
    for r, c_ in zip(row_ind, col_ind):
        if cost[r, c_] < np.inf:
            matches[r] = Match(
                obs_index=r,
                landmark_id=candidates[c_].id,
            )

    return matches
