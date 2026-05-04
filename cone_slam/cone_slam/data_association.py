"""Data association: match this scan's cone observations to existing
landmarks (or flag as new).

Originally followed AMZ §3.2 (Kabzan et al. arXiv:1905.05150):
    color gate (hard) → Mahalanobis gate → Hungarian assignment

The colour hard-gate is gone (#269). Why: the upstream classifier
(`color_classifier.classify(body_y, height)`) is a pose-relative
heuristic — it tags a cone YELLOW when it's at body_y < -0.5 and
BLUE when body_y > +0.5 at the moment of observation. When the car
yaws through a corner, the body_y of the same physical cone flips
sign within a few scans, so the new observations come in tagged with
the *opposite* colour from the existing landmark (which is locked to
its first-observation colour). The colour-bucket DA could not bridge
that gap → all observations marked NEW → cascade-detector kicked in →
SLAM rejected the whole scan. Live: at the second hairpin in
test_submodule, yaw rotated ~64° in 1 second, the entire SLAM_OBS
distribution flipped from B-dominated to Y-dominated, and DA went
from 100 % association to 0 %.

Replacement: position-based gating with **per-pair distance gate
that's tighter for cross-colour matches**. Same physical cone with a
flipped colour tag has 0 m offset, well within any cross-colour gate.
Two *physically distinct* cones of different colour are ≥ 3 m apart in
FS corridors, well outside the cross-colour gate. Concretely:

  - Same colour: full DISTANCE_GATE_M (2 m).
  - Different colour: CROSS_COLOR_DISTANCE_GATE_M (1 m).
  - Mahalanobis χ² applies in both cases as the secondary gate.
  - One Hungarian assignment over the combined cost matrix.

This preserves the start-grid wrong-match safety we tried to keep
when the fully-colour-blind variant got too greedy in cluster regions
(yellow cones within 2 m of orange/big-orange cones at the start
gate, observed live as a 4.6 m drift).

The data flow per scan:

    /Conos_raw  →  classify color per obs  →  this module  →  for each
                                                              matched obs:
                                                                add factor
                                                              for each
                                                              unmatched:
                                                                create
                                                                landmark
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from cone_slam.color_classifier import ConeColor
from cone_slam.landmark_db import Landmark, LandmarkDb


# Distance gate: an observation farther than this from a landmark's
# predicted body-frame position will not be considered the same cone.
# 1.0 m tight (vs the original 2 m) — at FS speeds (≤4 m/s) and
# ≤30°/s yaw, sequential observations of the same cone shift in body
# frame by <1 m between scans. Anything farther is more likely a
# different cone of the same color than a mis-prediction. Setting it
# tighter trades some "missed associations" (the obs becomes a NEW
# landmark instead of attaching to the right one) for far fewer
# WRONG associations. Wrong associations are the catastrophic-
# divergence mode we hit on 2026-04-27 with gate=2.0 — a single bad
# match in a sharp turn snaps iSAM2 to a 70°-off pose and cascades.
#
# Re-widened to 2.0 m on 2026-04-28 after fixing the double-publisher
# bug in tools/replay.sh — that bug had been dropping half the cone
# observations, masking how often DA was failing for legitimate reasons
# (predicted pose drifted past 1 m on the back stretch). With
# observations now flowing at full rate AND a velocity prior keeping
# the prediction tighter than before, 2.0 m gives the right backstop
# alongside Mahalanobis: physical cones are ~3 m apart so any wider
# match is wrong regardless of how loose iSAM2 claims the covariance is.
DISTANCE_GATE_M = 2.0


# Cross-colour Euclidean gate (#269). Tighter than DISTANCE_GATE_M so
# that cones of different colour can only associate when they're
# *almost exactly* on top of each other — i.e. the same physical cone
# with a flipped colour tag, not two adjacent cones from opposite
# corridor sides.
#
# 1.0 m sized to:
#   - ALLOW: same physical cone with a flipped tag (offset ≈ centroid
#     noise, typically < 0.2 m at FS ranges, plus per-scan SLAM jitter
#     of ~0.3-0.5 m on tight pose updates → max realistic offset ≈ 0.7 m,
#     comfortably under 1.0 m).
#   - REJECT: cross-corridor cones of different colour. FS layouts use
#     corridor widths ≥ 3 m, so distinct cones of opposite colour are
#     never within 1 m of each other.
#   - REJECT: same-side adjacent cones of different colour. Within-side
#     cone-pair spacing is ~3 m, well outside the gate.
CROSS_COLOR_DISTANCE_GATE_M = 1.0


# Time-since-association gate expansion was tested on 2026-04-29 in
# two variants (cap 1.5×, cap 4.0×). Both passed the iter-15
# regression but made the cascade WORSE (80 s drift 5.8–11.9 m vs
# A-alone's 2.1 m): once the gate opens past 3 m the optimizer
# starts matching cones from adjacent track sections, accelerating
# rather than recovering from the cascade. The mechanism only helps
# if pose drift during a rejection burst is the *dominant* cause of
# DA failure — but on cone-only LiDAR scenes the cascade is more
# often dominated by wrong-match-when-gate-opens. Disabled.
# Kept the current_step plumbing in associate() so re-enabling is a
# threshold change.
GATE_EXPANSION_PER_SCAN = 0.0    # disabled (no expansion)
GATE_EXPANSION_CAP      = 1.0    # static gate


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
#
# Effective gate radius for typical regimes (with χ²=9.21):
#   tight lm + tight pose:  σ_eff ≈ 0.55 m → gate ≈ 1.66 m
#   loose lm + tight pose:  σ_eff ≈ 1.05 m → gate ≈ 3.18 m  (capped to
#                                              DISTANCE_GATE_M=2.0 m)
#   tight lm + loose pose:  σ_eff ≈ 0.85 m → gate ≈ 2.58 m
POSE_COV_INFLATION    = 16.0  # multiplier on iSAM2's pose marginal
LANDMARK_COV_INFLATION = 16.0  # multiplier on iSAM2's landmark marginal
# Per-axis variance floor added to Σ_innov. (0.7 m)² = 0.49 m² so even
# a perfectly-constrained landmark/pose pair retains a ~0.7 m σ_eff.
COV_FLOOR_VAR_M2      = 0.49


@dataclass
class Observation:
    """One cone observation in body frame, post-color-classification.

    `sigma_xy` is the detector-reported position uncertainty in metres
    (centroid SE scaled by sqrt(N) and range). A non-positive value is
    a sentinel meaning "detector didn't report σ — fall back to the
    SLAM-side range-only formula".
    """

    body_x: float
    body_y: float
    height: float
    color: ConeColor
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

    Algorithm (#269 — colour as soft preference, not hard gate):
      1. Project every landmark into body frame.
      2. Build one (n_obs × n_lm) cost matrix:
         - Per-pair Euclidean offset.
         - Reject pairs above their colour-aware gate:
              same colour    → DISTANCE_GATE_M           (2 m)
              different col. → CROSS_COLOR_DISTANCE_GATE_M (1 m)
         - Reject pairs that fail the Mahalanobis χ² gate.
         - Surviving pairs get a finite Euclidean cost.
      3. One Hungarian assignment over the whole cost matrix.
      4. Matches with finite cost get a landmark_id; the rest get -1.

    `landmark_covariance_fn(id) -> 3×3 ndarray | None` is optional. If
    None or it returns None for a given id, we fall back to a default
    landmark σ (still combined with σ_obs for the gate).

    Args:
        observations: list of cone observations in body frame.
        pose_x, pose_y, pose_yaw: current pose estimate (2D, world frame).
        db: landmark database.
        landmark_covariance_fn: optional callback for the marginal
            covariance of L(id) (typically FactorGraph.landmark_covariance).

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
        # No landmarks yet — every obs is new.
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
        # Per-landmark gate expansion based on staleness (currently
        # disabled via GATE_EXPANSION_PER_SCAN=0 — see config block above).
        if current_step >= 0:
            stale = max(0, current_step - lm.last_seen_step)
            gate_mult = min(
                GATE_EXPANSION_CAP,
                1.0 + GATE_EXPANSION_PER_SCAN * stale)
        else:
            gate_mult = 1.0

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

            # Colour-aware Euclidean gate (#269).
            if o.color == lm.color:
                gate_eff = DISTANCE_GATE_M * gate_mult
            else:
                gate_eff = CROSS_COLOR_DISTANCE_GATE_M * gate_mult
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
        # else: gated out, stays as new (-1)

    return matches
