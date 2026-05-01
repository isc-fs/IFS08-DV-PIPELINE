"""Pure Pursuit lateral controller.

Geometric path follower: pick a target point on the path at lookahead
distance Ld ahead of the rear axle, draw an arc that reaches it, command the
steering angle that produces that arc.

Curvature of the chasing arc:
    κ = 2·sin(α) / Ld
    δ = atan(L · κ)             # kinematic bicycle, L = wheelbase
    steer_norm = δ / max_steer

where α is the angle from the vehicle heading to the target point.

Adaptive lookahead: Ld = clamp(k · v + L_min, L_min, L_max). Constant L_min
floors it at low speed (lookahead never collapses to zero), velocity term
stretches it on straights to smooth out steering.

Why Pure Pursuit (vs Stanley) for FS:
  - Stable at v→0 (no `softening_gain` denominator hack)
  - Naturally smooth — single geometric quantity, no PD on heading error
  - Cone-corridor following at 1–10 m/s is squarely Pure Pursuit territory

Reference frame: Path is published in `odom`, VehicleState pose is in `odom`.
We pick the target on the path geometrically (closest forward point past Ld
arc-length from the projected rear axle) and convert to body-frame heading.
"""
from __future__ import annotations
import math

from control.controllers.base import LateralController
from control.state import VehicleState
from control.reference import ReferenceTrajectory
from control.models.bicycle import KinematicBicycle


class PurePursuit(LateralController):
    def __init__(self,
                 lookahead_min: float = 1.5,
                 lookahead_k: float = 0.5,
                 lookahead_max: float = 8.0,
                 model: KinematicBicycle | None = None):
        self.lookahead_min = lookahead_min
        self.lookahead_k = lookahead_k
        self.lookahead_max = lookahead_max
        self.model = model or KinematicBicycle()

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        if ref.empty:
            return 0.0

        # Adaptive lookahead. Use scalar speed, not vx, so reverse motion
        # doesn't shrink Ld below the floor.
        Ld = max(self.lookahead_min,
                 min(self.lookahead_max,
                     self.lookahead_min + self.lookahead_k * state.speed))

        # Find the index of the path point closest to the car (in odom frame).
        # This anchors us to where we are on the path; we then walk forward by
        # arc length Ld to pick the target. Doing a fresh nearest-search every
        # tick avoids the "anchor stuck" failure mode of the old controller —
        # there is no per-tick state carried between calls.
        nearest = _nearest_index(state.x, state.y, ref.x, ref.y)

        # Target = first path point at least Ld of arc length past `nearest`.
        # Falls back to the last point if Ld would walk past path end —
        # geometrically still valid, just a shorter chasing arc.
        s_anchor = ref.s[nearest]
        target_idx = nearest
        for i in range(nearest, len(ref.s)):
            if ref.s[i] - s_anchor >= Ld:
                target_idx = i
                break
        else:
            target_idx = len(ref.x) - 1

        tx, ty = ref.x[target_idx], ref.y[target_idx]

        # Body-frame target: rotate the world-frame offset (tx-x, ty-y) by -yaw
        # to express it relative to the car's heading. body_x is forward,
        # body_y is left.
        dx = tx - state.x
        dy = ty - state.y
        cos_y, sin_y = math.cos(state.yaw), math.sin(state.yaw)
        body_x =  cos_y * dx + sin_y * dy
        body_y = -sin_y * dx + cos_y * dy

        # If the target is behind us (shouldn't happen with nearest+arc logic
        # but possible at sharp planner replans), reflect to forward to avoid
        # the tan(α) singularity at α = ±π/2.
        if body_x <= 1e-3:
            return 0.0

        # Pure Pursuit curvature: κ = 2·sin(α)/Ld where α = atan2(body_y,
        # body_x). The 2·body_y / |target|² form is identical and avoids the
        # atan2+sin pair.
        target_dist_sq = body_x * body_x + body_y * body_y
        if target_dist_sq < 1e-6:
            return 0.0
        kappa = 2.0 * body_y / target_dist_sq

        steer_rad = self.model.steer_for_curvature(kappa)
        return self.model.normalize_steer(steer_rad)


def _nearest_index(x: float, y: float, xs, ys) -> int:
    """Return index of the path point closest to (x, y). Linear scan — paths
    are 30-ish points, no need for spatial indexing."""
    best_i = 0
    best_d2 = float("inf")
    for i, (px, py) in enumerate(zip(xs, ys)):
        dx, dy = px - x, py - y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i
