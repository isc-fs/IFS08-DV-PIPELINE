"""Pure Pursuit lateral controller.

Geometric path follower: pick a target point on the path at lookahead
distance Ld ahead of the rear axle, draw an arc that reaches it, command the
steering angle that produces that arc.

Curvature of the chasing arc:
    κ = 2·sin(α) / Ld
    δ = atan(L · κ)             # kinematic bicycle, L = wheelbase
    steer_norm = δ / max_steer

where α is the angle from the vehicle heading to the target point.

Adaptive lookahead: Ld = clamp(L_min + k·v, L_min, min(L_max, β·R_local)).
Constant L_min floors it at low speed (lookahead never collapses to zero),
velocity term stretches it on straights, and the β·R_local cap keeps it
below the local turn radius — Pure Pursuit's chase target must land *inside*
the curve, not past the apex on the next straight (#260). β ≈ 0.7 is the
standard Pure Pursuit rule of thumb. R_local = 1 / max|κ| over the next
lookahead window of the path; we read κ from `ref.curvature` rather than
recomputing.

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
                 # Pure Pursuit theory: chase target must land inside the
                 # curve, i.e. Ld ≤ β·R for some β < 1. β=0.7 is a common
                 # FS-Driverless and ground-vehicle Pure Pursuit value
                 # (e.g. Coulter '92 §4 recommends Ld < radius; AMZ /
                 # MIT FSAE references use 0.6–0.8). On a 4 m hairpin
                 # this caps Ld at 2.8 m vs the previous unconstrained
                 # 4 m at 5 m/s — keeps the controller from chasing a
                 # past-apex target through the corner. (#260)
                 lookahead_radius_factor: float = 0.7,
                 model: KinematicBicycle | None = None):
        self.lookahead_min = lookahead_min
        self.lookahead_k = lookahead_k
        self.lookahead_max = lookahead_max
        self.lookahead_radius_factor = lookahead_radius_factor
        self.model = model or KinematicBicycle()

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        if ref.empty:
            return 0.0

        # Adaptive lookahead. Use scalar speed, not vx, so reverse motion
        # doesn't shrink Ld below the floor. The R_local cap is applied
        # below once we know what's ahead on the path.
        Ld = max(self.lookahead_min,
                 min(self.lookahead_max,
                     self.lookahead_min + self.lookahead_k * state.speed))

        # Curvature-adaptive cap (#260). Walk forward from `nearest`
        # along the path and find the tightest κ within the *current*
        # Ld window — that's the corner the controller is about to
        # enter. Cap Ld at β·R_local so the chase target stays inside
        # the curve. Doing this *before* picking target_idx means the
        # target search uses the capped Ld directly; no second pass.
        nearest_for_cap = _nearest_index(state.x, state.y, ref.x, ref.y)
        kappa_max = _max_kappa_in_window(ref, nearest_for_cap, Ld)
        if kappa_max > 1e-3:
            radius_cap = self.lookahead_radius_factor / kappa_max
            Ld = min(Ld, radius_cap)
            # And re-floor at L_min — even on a hairpin we must look
            # at least L_min ahead, otherwise Pure Pursuit destabilises
            # at low speed (the chase target collapses onto the car).
            Ld = max(Ld, self.lookahead_min)

        # Anchor to where we are on the path. Reuse the nearest-index
        # we computed for the radius cap above.
        nearest = nearest_for_cap

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


def _max_kappa_in_window(ref: ReferenceTrajectory, nearest: int,
                         window_m: float) -> float:
    """Maximum |κ| from `nearest` over `window_m` of arc length ahead.

    Reads the path's pre-computed curvature from `ref.curvature` —
    populated by `ReferenceTrajectory.from_xy` (controller side, finite
    differences over the path xy). Returns 0.0 on a path with no
    curvature samples (degenerate / very short).
    """
    if not ref.curvature:
        return 0.0
    if nearest >= len(ref.curvature):
        return 0.0
    s_anchor = ref.s[nearest]
    kmax = 0.0
    for i in range(nearest, len(ref.s)):
        if ref.s[i] - s_anchor > window_m:
            break
        k = abs(ref.curvature[i])
        if k > kmax:
            kmax = k
    return kmax


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
