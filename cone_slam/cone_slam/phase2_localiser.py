"""Phase 2 localisation: pose-only EKF against a frozen cone map.

Phase 1 hands a `FrozenMap` to Phase 2 at lap completion. From that
point onward we stop mapping and start correcting pose drift via
cone observations matched to the immutable map. The state is just
the SE(2) pose `[x, y, theta]` — there's no landmark state in the
filter, which is what makes this structurally robust: no DA cascade
can corrupt the map because the map is frozen.

The data-association loop is Mahalanobis-gated nearest-landmark:
each body-frame observation is projected into the world frame using
the predicted pose, the FrozenMap's KD-tree returns the nearest
landmark, and the Mahalanobis distance of the innovation under the
current uncertainty is checked against the χ² gate. Gated matches
become sequential EKF updates.

Pure Python; no ROS dependencies. Drives are deterministic, so the
regression suite can exercise this module against a recorded bag
the same way Phase 1 does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cone_slam.frozen_map import FrozenMap
from cone_slam.phase1_mapper import Observation, Pose2D


# 2-DOF chi-squared 95% gate. Innovations live in R² (body-frame
# observation residual), so this is the right table value.
CHI2_95_2DOF: float = 5.991


@dataclass
class UpdateSummary:
    """Caller-facing summary of a single `update()` call.

    `n_matched` counts gated matches that were folded into the
    state; `n_gated_out` counts observations that found a candidate
    landmark but failed the Mahalanobis test; `n_unmatched` counts
    observations with no candidate within `max_match_radius_m`.
    """
    n_obs: int = 0
    n_matched: int = 0
    n_gated_out: int = 0
    n_unmatched: int = 0
    mean_innovation_m: float = 0.0


class Phase2Localiser:
    """Pose-only EKF over a frozen cone map.

    State `x = [x, y, theta]` lives in the world frame (same frame
    the map was built in). Covariance `P` is 3×3.

    Parameters
    ----------
    frozen_map
        Snapshot from Phase 1. Stays immutable for the lifetime of
        the localiser.
    init_pose
        Best-guess pose at hand-off — typically the SLAM pose at
        the moment the lap detector fired.
    init_cov_diag
        Diagonal of P at init. Defaults to (0.5 m, 0.5 m, 5°). Set
        smaller if you trust the hand-off; larger if you don't.
    obs_sigma_m
        Observation noise σ per axis (assumed isotropic in body
        frame). Same scale as Phase 1's per-cone σ.
    max_match_radius_m
        Hard cap on candidate selection. The KD-tree returns the
        nearest landmark within this radius; anything farther is
        treated as unmatched. Keeps DA cheap and rejects gross
        outliers before the Mahalanobis math runs.
    mahalanobis_gate
        χ² gate for the 2-DOF innovation. Default is 95% (5.991);
        tighten for stricter outlier rejection.
    """

    def __init__(
        self,
        frozen_map: FrozenMap,
        init_pose: Pose2D,
        *,
        init_cov_diag: tuple[float, float, float] = (0.5, 0.5, math.radians(5.0)),
        obs_sigma_m: float = 0.20,
        max_match_radius_m: float = 3.0,
        mahalanobis_gate: float = CHI2_95_2DOF,
    ) -> None:
        self._map = frozen_map
        self._x = np.array([init_pose.x, init_pose.y, init_pose.yaw],
                           dtype=float)
        self._P = np.diag(np.asarray(init_cov_diag, dtype=float) ** 2)
        self._obs_sigma = float(obs_sigma_m)
        self._max_match_radius = float(max_match_radius_m)
        self._gate = float(mahalanobis_gate)

    # ----- accessors -----

    @property
    def pose(self) -> Pose2D:
        return Pose2D(float(self._x[0]), float(self._x[1]),
                      float(self._x[2]))

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def map(self) -> FrozenMap:
        return self._map

    # ----- predict step -----

    def predict(
        self,
        dx_body: float,
        dy_body: float,
        dtheta: float,
        *,
        sigma_xy: float = 0.05,
        sigma_yaw: float = math.radians(0.5),
    ) -> None:
        """Advance the state by a body-frame motion delta.

        Inputs are increments per cone-scan tick (not per unit time)
        so the caller integrates `/odom` velocity and passes the
        result. Process noise σ values are applied as additive
        diagonal Q — small per-tick but accumulate when the filter
        runs without measurement updates.

        Jacobian F follows the standard SE(2) prediction:
            x' = x + dx_body·cos(θ) − dy_body·sin(θ)
            y' = y + dx_body·sin(θ) + dy_body·cos(θ)
            θ' = θ + dθ
        """
        theta = self._x[2]
        c, s = math.cos(theta), math.sin(theta)

        self._x[0] += dx_body * c - dy_body * s
        self._x[1] += dx_body * s + dy_body * c
        self._x[2] = _wrap_pi(self._x[2] + dtheta)

        # F = ∂f/∂x. Only θ-dependence matters here.
        F = np.array([
            [1.0, 0.0, -dx_body * s - dy_body * c],
            [0.0, 1.0,  dx_body * c - dy_body * s],
            [0.0, 0.0,  1.0],
        ])
        Q = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
        self._P = F @ self._P @ F.T + Q

    # ----- update step -----

    def update(self, observations: list[Observation]) -> UpdateSummary:
        """Sequentially fold gated cone matches into the state.

        Each observation is associated to at most one frozen
        landmark (the KD-tree nearest within `max_match_radius_m`).
        The Mahalanobis distance of the resulting innovation is
        checked against `self._gate`; passing matches drive a
        standard EKF update. Sequential updates are mathematically
        equivalent to a batch update when the per-observation
        noise is uncorrelated, which it is for cone detections.
        """
        summary = UpdateSummary(n_obs=len(observations))
        if not observations or len(self._map) == 0:
            return summary

        innovations: list[float] = []
        for obs in observations:
            zx, zy = float(obs.body_x), float(obs.body_y)

            # Project the observation into the world frame using
            # the current pose estimate. The nearest landmark to
            # *that point* is our candidate — DA happens in world
            # frame, not body frame, so it stays valid as the pose
            # corrects mid-scan.
            theta = self._x[2]
            c, s = math.cos(theta), math.sin(theta)
            world_x = self._x[0] + zx * c - zy * s
            world_y = self._x[1] + zx * s + zy * c

            idx, dist = self._map.query_nearest(
                np.array([world_x, world_y]),
                max_distance=self._max_match_radius,
            )
            if idx < 0:
                summary.n_unmatched += 1
                continue

            lm_x, lm_y = self._map.positions[idx]

            # Predicted observation: project the landmark back into
            # body frame using the current pose.
            dxw = lm_x - self._x[0]
            dyw = lm_y - self._x[1]
            hx = dxw * c + dyw * s
            hy = -dxw * s + dyw * c

            # Innovation and observation Jacobian.
            y_innov = np.array([zx - hx, zy - hy])
            H = np.array([
                [-c, -s,  hy],
                [ s, -c, -hx],
            ])

            # Per-cone obs noise + per-landmark sigma (some map
            # cones were observed many times in Phase 1; we trust
            # those positions more, hence smaller R).
            lm_sigma = float(self._map.sigmas[idx])
            r_var = self._obs_sigma ** 2 + lm_sigma ** 2
            R = r_var * np.eye(2)
            S = H @ self._P @ H.T + R

            # Mahalanobis gate.
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                summary.n_gated_out += 1
                continue
            mahal = float(y_innov @ Sinv @ y_innov)
            if mahal > self._gate:
                summary.n_gated_out += 1
                continue

            # Kalman gain + state / cov update.
            K = self._P @ H.T @ Sinv
            self._x = self._x + K @ y_innov
            self._x[2] = _wrap_pi(self._x[2])

            # Joseph form would be more numerically stable for
            # repeated updates, but the standard form is fine for
            # the modest number of cones per scan we see and avoids
            # the extra matrix multiplies. Switch if covariance
            # ever loses symmetry in practice.
            I3 = np.eye(3)
            self._P = (I3 - K @ H) @ self._P
            # Re-symmetrise to fight floating-point drift.
            self._P = 0.5 * (self._P + self._P.T)

            summary.n_matched += 1
            innovations.append(float(np.linalg.norm(y_innov)))

        if innovations:
            summary.mean_innovation_m = float(np.mean(innovations))
        return summary


def _wrap_pi(angle: float) -> float:
    """Wrap to (-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
