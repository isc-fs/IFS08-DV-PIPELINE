"""Stanley lateral controller.

Geometric path follower that projects the front axle onto the path and
combines two terms:

    δ = ψ_e + atan(k · e_y / (k_soft + v)) - k_yaw_rate · ω_ref

where:
    ψ_e             heading error between path tangent and vehicle yaw,
                    wrapped to [-π, π]
    e_y             signed cross-track error at the front axle (m,
                    positive = path is to the left of the car)
    v               longitudinal speed (m/s)
    k               cross-track gain (1/s scaling)
    k_soft          softening term (m/s) — keeps the denominator
                    bounded at v→0, preventing the cross-track term
                    from saturating the actuator on a stopped car
    k_yaw_rate      damping against commanded yaw rate (rad / (rad/s));
                    0.0 disables the term
    ω_ref           yaw rate the bicycle would produce at the current
                    speed and previous steering — used only for damping

Reference: Hoffmann et al., "Autonomous Automobile Trajectory
Tracking for Off-Road Driving" (Stanley, 2007),
https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
Port adapted from alt_pipeline/control/control/controlador_stanley.py
onto the LateralController ABC used by control_node.

Sign convention matches the rest of `controllers/`: positive δ → LEFT
(math convention). control_node._tick flips the sign at the actuator
boundary to match UE5/Chaos automotive convention.

Why Stanley alongside Pure Pursuit:
  - Sharper response on constant-curvature paths (skidpad figure-8) —
    cross-track and heading terms have well-defined signs across the
    full circle, no chase-target geometry to tune.
  - Picks the nearest projection rather than a forward chase target,
    so on a tight hairpin it doesn't risk chasing a past-apex
    point. Pure Pursuit needs the β·R radius cap (#260 follow-up)
    to avoid that; Stanley doesn't.
  - At v→0 the softening_gain holds the cross-track term stable; the
    Pure Pursuit radius cap also holds in that regime, so both are
    safe for tight low-speed starts.

When NOT to use Stanley:
  - On near-straight paths with noisy planner yaw, the heading term
    can dominate and produce small oscillations. Acceleration mission
    keeps Pure Pursuit for this reason.
"""
from __future__ import annotations
import math

from control.controllers.base import LateralController
from control.state import VehicleState
from control.reference import ReferenceTrajectory
from control.models.bicycle import KinematicBicycle


class Stanley(LateralController):
    """Stanley lateral controller behind the LateralController ABC."""

    def __init__(
        self,
        k: float = 4.0,
        k_soft: float = 6.0,
        k_yaw_rate: float = 0.0,
        k_damp_steer: float = 0.0,
        model: KinematicBicycle | None = None,
    ) -> None:
        self.k = k
        self.k_soft = k_soft
        self.k_yaw_rate = k_yaw_rate
        # Steering-delay damping: small fraction of the previous tick's
        # commanded steering is subtracted from the new command. Pure
        # additive smoothing — 0.0 disables, ~0.5 visibly damps. The
        # actuator slew limiter in control_node._tick handles the
        # rate-of-change side; this term reduces single-tick swings on
        # noisy paths.
        self.k_damp_steer = k_damp_steer
        self.model = model or KinematicBicycle()
        # Previous tick's normalized steering — input to the damping
        # term. Reset() clears it.
        self._prev_steer_norm: float = 0.0

    def reset(self) -> None:
        self._prev_steer_norm = 0.0

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        if ref.empty:
            return 0.0

        # Project the front axle onto the path. Stanley operates on
        # the front axle (not the CG/rear axle Pure Pursuit uses)
        # because the steering authority is at the front; the
        # geometric error there is what the controller can directly
        # null out.
        fx, fy = self.model.front_axle_xy(state.x, state.y, state.yaw)

        target_idx, dx, dy, abs_err = _nearest_projection(fx, fy, ref.x, ref.y)

        # Heading error: path tangent minus vehicle yaw, wrapped to
        # [-π, π]. ref.yaw is populated by ReferenceTrajectory.from_xy
        # via forward-difference of consecutive waypoints — small bias
        # near the path tail (last yaw copies the previous one) but
        # accurate everywhere else.
        path_yaw = ref.yaw[target_idx] if ref.yaw else state.yaw
        yaw_error = _wrap_pi(path_yaw - state.yaw)

        # Signed cross-track error. The dot product with the
        # front-axle "right-hand" vector (sin yaw, -cos yaw) yields
        # positive when the path is to the RIGHT of the car (i.e.
        # the offset vector projects onto +right), and we flip the
        # sign so positive = path-is-LEFT, matching the math
        # convention used by the rest of `controllers/` (positive
        # steering = LEFT). Result: a positive crosstrack error
        # asks the controller to steer LEFT, which is the same sign
        # as a positive yaw_error.
        right = (math.sin(state.yaw), -math.cos(state.yaw))
        proj = dx * right[0] + dy * right[1]
        crosstrack = -math.copysign(abs_err, proj)

        # Cross-track term — atan keeps the contribution bounded
        # below ±π/2 even when (k_soft + v) is small.
        cte_term = math.atan2(self.k * crosstrack, self.k_soft + state.speed)

        # Yaw-rate damping (Hoffmann §3.3). ω_ref ≈ v · sin(δ_prev) / L
        # from the kinematic bicycle. Subtracts a velocity-scaled
        # multiple of the last commanded yaw rate from the new
        # steering command. Disabled (0.0) by default; turn on if the
        # car wags at high speed.
        prev_steer_rad = self._prev_steer_norm * self.model.max_steer_rad
        omega_ref = state.speed * math.sin(prev_steer_rad) / self.model.wheelbase
        yaw_rate_term = self.k_yaw_rate * omega_ref

        steer_rad = yaw_error + cte_term - yaw_rate_term

        # Optional single-tick damping against the previous command.
        # k_damp_steer is dimensionless; 0.0 = off, 0.5 = blends in
        # half of the previous tick's steering as inertia. Helps on
        # noisy planner yaw.
        if self.k_damp_steer > 0.0:
            steer_rad -= self.k_damp_steer * (steer_rad - prev_steer_rad)

        # Clamp and convert to normalized [-1, 1]. The actuator slew
        # limiter in control_node._tick handles rate-of-change at the
        # publish boundary; this just bounds the steady-state command.
        steer_rad = max(
            -self.model.max_steer_rad, min(self.model.max_steer_rad, steer_rad)
        )
        steer_norm = self.model.normalize_steer(steer_rad)
        self._prev_steer_norm = steer_norm
        return steer_norm


def _nearest_projection(
    x: float, y: float, xs, ys,
) -> tuple[int, float, float, float]:
    """Return (idx, dx, dy, dist) of the nearest path point to (x, y).

    Linear scan — paths are 30-ish points, no spatial index needed.
    Matches the convention used by the alt_pipeline Stanley port: dx,
    dy are the offsets FROM the front axle TO the path point.
    """
    best_i = 0
    best_d2 = float("inf")
    best_dx = 0.0
    best_dy = 0.0
    for i, (px, py) in enumerate(zip(xs, ys)):
        ddx = px - x
        ddy = py - y
        d2 = ddx * ddx + ddy * ddy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_dx = ddx
            best_dy = ddy
    return best_i, best_dx, best_dy, math.sqrt(best_d2)


def _wrap_pi(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
