"""Data association: match this scan's cone observations to existing
landmarks (or flag as new).

Per AMZ §3.2 (Kabzan et al. arXiv:1905.05150) the recipe is:
    color gate → Mahalanobis gate → Hungarian assignment

`associate()` accepts a `landmark_covariance_fn(id) -> 3×3 ndarray | None`
callback. When provided, each candidate's Mahalanobis distance is
computed in the body frame using:

    Σ_innov_body = R_world→body · Σ_landmark_world · R_world→body^T
                   + σ_obs² · I

and gated by the χ² threshold for 2 DOF (default 99 % = 9.21). When
the covariance isn't available yet (e.g. a landmark just created on
the same scan, before iSAM2 marginalizes it), we fall back to the
Euclidean DISTANCE_GATE_M backstop. The Euclidean gate also stays
active as a sanity cap regardless of Mahalanobis: two physical cones
are at least ~3 m apart, so any match across more than DISTANCE_GATE_M
is almost certainly a wrong-cone match no matter how loose the
landmark's covariance might claim.

The data flow per scan:

    /Conos_raw  →  classify color per obs  →  this module  →  for each
                                                              matched obs:
                                                                add factor
                                                              for each
                                                              unmatched:
                                                                create
                                                                landmark

Color gating cuts the search 4× (FS cones come in 4 classes) before
the Mahalanobis distance is even computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
) -> List[Match]:
    """Match observations to landmarks. One Match per observation.

    Algorithm:
      1. Bucket landmarks by color.
      2. For each color bucket:
         - project landmarks into body frame using current pose
         - for each (obs, landmark) pair:
              · compute innovation in body frame (obs − lm_body)
              · compute innovation covariance in body frame
                  Σ = R_world→body · Σ_lm_world · R_world→body^T
                      + σ_obs² · I
              · gate by Mahalanobis χ² (≤ MAHALANOBIS_CHI2)
              · also gate by Euclidean (≤ DISTANCE_GATE_M) as a hard
                physical-cone-spacing backstop
         - Hungarian assignment minimizes total χ² cost
         - matches with finite cost get a landmark_id; the rest get -1
      3. Concatenate per-color results.

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

    # Bucket observation indices by color.
    by_color: Dict[ConeColor, List[int]] = {}
    for i, o in enumerate(observations):
        by_color.setdefault(o.color, []).append(i)

    # World→body rotation: if R_b2w = [[c, -s], [s, c]] (yaw), then
    # R_w2b = R_b2w^T = [[c, s], [-s, c]].
    c_yaw = float(np.cos(pose_yaw))
    s_yaw = float(np.sin(pose_yaw))
    R_w2b = np.array([[ c_yaw, s_yaw],
                      [-s_yaw, c_yaw]])

    default_lm_cov_body = (DEFAULT_LANDMARK_SIGMA_M ** 2) * np.eye(2)
    default_obs_var = DEFAULT_OBS_SIGMA_M ** 2

    for color, obs_indices in by_color.items():
        candidates = db.all_by_color(color)
        if not candidates:
            # No landmarks of this color exist yet — every obs is new.
            continue

        n_obs = len(obs_indices)
        n_lm = len(candidates)
        cost = np.full((n_obs, n_lm), np.inf)

        # Pre-fetch landmark body-frame positions and (optionally)
        # body-frame covariances. The covariance call is potentially
        # expensive (iSAM2 marginal computation) — so we do it once
        # per landmark, not per (obs × landmark) pair.
        lm_bodies = np.empty((n_lm, 2))
        lm_cov_body = [default_lm_cov_body] * n_lm
        for j, lm in enumerate(candidates):
            lm_bodies[j] = _world_to_body(pose_x, pose_y, pose_yaw,
                                          lm.position[:2])
            if landmark_covariance_fn is not None:
                cov_world = landmark_covariance_fn(lm.id)
                if cov_world is not None:
                    sigma_world_xy = cov_world[:2, :2]
                    lm_cov_body[j] = R_w2b @ sigma_world_xy @ R_w2b.T

        for j in range(n_lm):
            lm_body = lm_bodies[j]
            sigma_lm = lm_cov_body[j]

            # Pose-uncertainty contribution to Σ_innov via the Jacobian
            # of body-frame projection w.r.t. (yaw, x_w, y_w). With
            #   lm_body = R_w2b (lm_world − p),   R_w2b = [[c, s],[-s, c]]
            # the partials are:
            #   ∂lm_body/∂yaw = [ lm_body_y, −lm_body_x]
            #   ∂lm_body/∂x_w = [−c, +s]
            #   ∂lm_body/∂y_w = [−s, −c]
            # Without this term the Mahalanobis gate collapses to ~0.3 m
            # for tightly-constrained landmarks even when the predicted
            # pose has drifted 1–2 m — exactly the regime where DA is
            # most needed. With it, uncertain pose → wider gate, certain
            # pose → tighter gate.
            if pose_xy_yaw_cov is not None:
                J = np.array([
                    [ lm_body[1], -c_yaw, -s_yaw],
                    [-lm_body[0],  s_yaw, -c_yaw],
                ])
                sigma_pose_contrib = J @ pose_xy_yaw_cov @ J.T
            else:
                sigma_pose_contrib = np.zeros((2, 2))

            for ii, obs_idx in enumerate(obs_indices):
                o = observations[obs_idx]
                dx = o.body_x - lm_body[0]
                dy = o.body_y - lm_body[1]
                d_eu = float(np.hypot(dx, dy))
                if d_eu > DISTANCE_GATE_M:
                    # Euclidean physical-spacing backstop — skip
                    # before doing the matrix math.
                    continue
                obs_var = (o.sigma_xy ** 2) if o.sigma_xy > 0 \
                    else default_obs_var
                sigma_innov = (sigma_lm
                               + obs_var * np.eye(2)
                               + sigma_pose_contrib)
                innov = np.array([dx, dy])
                try:
                    sol = np.linalg.solve(sigma_innov, innov)
                except np.linalg.LinAlgError:
                    continue
                d2 = float(innov @ sol)
                if d2 <= MAHALANOBIS_CHI2:
                    cost[ii, j] = d2

        # Hungarian doesn't accept +∞; replace with a large finite value.
        big = 1e6
        cost_solver = np.where(np.isinf(cost), big, cost)

        row_ind, col_ind = linear_sum_assignment(cost_solver)
        for r, c_ in zip(row_ind, col_ind):
            if cost[r, c_] < np.inf:
                obs_idx = obs_indices[r]
                matches[obs_idx] = Match(
                    obs_index=obs_idx,
                    landmark_id=candidates[c_].id,
                )
            # else: gated out, stays as new (-1)

    return matches
