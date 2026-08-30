"""PI velocity tracker with curvature feedforward + kinematic stop cap.

Two stages each tick:

1. SETPOINT — derive a target speed from path geometry + stop state:
       v_set = min(
           v_max,                                  # global cap
           sqrt(R · a_lat_max),                    # cornering grip limit
           sqrt(2 · a_dec_max · d_to_stop),        # kinematic decel cap
       )
   R is the inverse of the maximum |κ| within the next ~3 s of path
   travel — using the local point's κ alone is too noisy. d_to_stop comes
   from the reference trajectory (set by the stop-detection logic in the
   ROS node, not in this controller).

2. TRACK — PI on (v_set - v):
       u = Kp · err + Ki · ∫err dt
   With a deadband around the setpoint to avoid bang-bang flip between
   throttle and regen on small errors. Anti-windup: integrator only updates
   when the unsaturated command would land in [-1, 1].

Output split:
   u >= +deadband  → throttle =  u, regen = 0
   u <= -deadband  → throttle = 0,  regen = -u
   |u| < deadband  → throttle = 0,  regen = 0    (coast)

Why no D term: at 40 Hz with SLAM-derived velocity (10 Hz update, smoothed
internally), the derivative is mostly quantisation noise. The audit's
"derivative-on-error not -on-measurement" failure mode (item #6) goes away
by simply not having a D term. Add it back if and when we have a clean
high-rate v_meas signal.
"""
from __future__ import annotations
import math
from typing import Tuple

from control.controllers.base import LongitudinalController
from control.state import VehicleState
from control.reference import ReferenceTrajectory


# Tick period — must match ControlNode.PUBLISH_RATE_HZ. Used only by the
# integrator. Hard-coded so this module doesn't need to know about the node.
_DT = 1.0 / 40.0


class PIVelocity(LongitudinalController):
    def __init__(self,
                 v_max: float = 12.0,
                 a_lat_max: float = 3.0,
                 a_dec_max: float = 4.0,
                 kp: float = 0.5,
                 ki: float = 0.05,
                 deadband: float = 0.2,
                 lookahead_curvature_s: float = 3.0,
                 throttle_max: float = 0.6,
                 v_meas_tau: float = 0.15,
                 v_meas_max_accel: float = 20.0):
        self.v_max = v_max
        self.a_lat_max = a_lat_max
        self.a_dec_max = a_dec_max
        self.kp = kp
        self.ki = ki
        self.deadband = deadband
        self.lookahead_curvature_s = lookahead_curvature_s
        # Throttle saturation. Lower than the regen cap because if SLAM's
        # velocity estimate lags reality (we observed this triggering at
        # ~6 m/s during baseline runs — pose froze, controller kept
        # commanding full throttle, real car ran away to ~24 m/s and
        # crashed), a tighter throttle ceiling caps how badly the
        # state-estimator divergence can blow up. Regen stays at 1.0 so
        # the controller can still demand max stopping power.
        self.throttle_max = throttle_max
        # v_meas conditioning. In the sim /odom.vx is clean (divergence from
        # true ground speed measured at std 0.024 m/s on a 200 s autocross
        # bag), so feeding it raw to kp is harmless. On the REAL car state.vx
        # is uDV dead-reckoning / GSS — noisier, and it occasionally emits a
        # spurious single-tick value (e.g. a 0 during a filter reset). Fed
        # raw to kp that becomes an intermittent full-throttle/regen SPIKE:
        # err jumps by ~v, u = kp·err saturates for a tick, then recovers.
        # Two-stage conditioning rejects it at the INPUT — the #306 output
        # slew can only delay a sustained spike, it cannot reject a glitch:
        #   1. plausibility clamp — reject |Δv| beyond v_meas_max_accel·dt,
        #      which no real vehicle can produce in one tick, so a glitch
        #      sample is clipped to the achievable bound. Set well ABOVE the
        #      car's true accel/decel envelope (~4-8 m/s²) so it only ever
        #      catches non-physical jumps, never real dynamics.
        #   2. EMA low-pass (τ = v_meas_tau) — smooth the residual noise the
        #      P-term would otherwise amplify straight into torque.
        # Loop gains are slow (kp=0.5, ki=0.05) so the ~0.15 s filter lag
        # costs negligible phase margin.
        self.v_meas_tau = v_meas_tau
        self.v_meas_max_accel = v_meas_max_accel
        self._v_filt: float | None = None
        self._integral = 0.0
        # Diagnostic side-channel — last computed values, for the
        # control node to publish on debug topics. Read these *after*
        # compute() returns; they're undefined before the first call.
        self.last_v_set: float = 0.0
        self.last_kappa_max: float = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._v_filt = None

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> Tuple[float, float]:
        v_set = self._setpoint(state, ref)
        # Stash for diagnostic publish (#260 follow-up).
        self.last_v_set = v_set
        # Track SIGNED body-frame forward velocity (not magnitude). Using
        # |v| would feed the wrong sign back if the car ever ended up
        # going backward (e.g. pushed by collision): |v| = 5 looks like
        # over-speed and the PI would issue regen, accelerating the
        # reverse — a positive-feedback loop. With signed vx, a reversing
        # car looks like negative velocity, so PI commands throttle to
        # decelerate the reverse and bring the car back to forward travel.
        # The sim-side single-quadrant regen fix (PR #160) ensures regen
        # commanded at vx ≤ 0 produces zero motor torque, so this can't
        # spiral even on transient reverse motion from collisions.
        return self._track(self._condition_velocity(state.vx), v_set)

    # ----------------------------------------------------- v_meas conditioning

    def _condition_velocity(self, v_raw: float) -> float:
        """Reject non-physical single-tick jumps, then EMA low-pass. See the
        __init__ rationale. Returns the raw sample on the first call (seeds
        the filter)."""
        if self._v_filt is None:
            self._v_filt = v_raw
            self._v_prev_raw = v_raw
            return v_raw
        # 1. Plausibility clamp, applied to the RAW stream — compared against
        #    the previous RAW sample, NOT against the filtered estimate. The
        #    filter legitimately lags during real acceleration, so clamping
        #    against it would read that lag as a jump, clip genuine dynamics,
        #    and compound its own lag (measured: 3.2 m/s error tracking a real
        #    6 m/s² decel, vs 0.9 m/s from the EMA alone).
        #    The CLAMPED value becomes the next reference: a one-sample glitch
        #    must not capture the reference and make the following (correct)
        #    sample look like a jump in the opposite direction.
        max_dv = self.v_meas_max_accel * _DT
        dv = v_raw - self._v_prev_raw
        if dv > max_dv:
            v_raw = self._v_prev_raw + max_dv
        elif dv < -max_dv:
            v_raw = self._v_prev_raw - max_dv
        self._v_prev_raw = v_raw
        # 2. EMA low-pass. alpha = dt / (tau + dt).
        alpha = _DT / (self.v_meas_tau + _DT)
        self._v_filt += alpha * (v_raw - self._v_filt)
        return self._v_filt

    # ------------------------------------------------------------- setpoint

    def _setpoint(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        # Look ahead by current_speed × horizon_s of arc length and find the
        # tightest corner in that window. Using the local κ alone is too
        # noisy (audit item #9); using the global max overshoots straights
        # adjacent to hairpins.
        kappa_max = self._max_curvature_ahead(state, ref)
        # Stash for diagnostic publish (#260 follow-up). Float NaN
        # would also work but Float32 publishers prefer real values.
        self.last_kappa_max = kappa_max
        if kappa_max < 1e-3:
            v_corner = self.v_max
        else:
            R = 1.0 / kappa_max
            v_corner = math.sqrt(R * self.a_lat_max)

        # Kinematic stop cap. ref.stop_distance is +inf when no stop is
        # latched, so v_stop = +inf and this cap is non-binding. Once latched
        # by the ROS node, v_stop pulls the setpoint smoothly to zero.
        if ref.stop_distance < float("inf") and ref.stop_distance > 0.0:
            v_stop = math.sqrt(2.0 * self.a_dec_max * ref.stop_distance)
        else:
            v_stop = float("inf")
        if ref.stop_latched and ref.stop_distance <= 0.0:
            v_stop = 0.0

        return max(0.0, min(self.v_max, v_corner, v_stop))

    def _max_curvature_ahead(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        """Maximum |κ| ahead of the car's nearest point, over the entire
        path.

        Originally windowed by `current_speed × horizon_s`, but with the
        path-output cap from PR #261 (12 m of arc length), the corner
        can't physically be further than the path is long. Scanning the
        full forward portion catches corners early — including hairpins
        whose curvature is concentrated at one or two points along an
        otherwise-straight approach (#260 follow-up).
        """
        if not ref.curvature:
            return 0.0
        nearest = _nearest_index(state.x, state.y, ref.x, ref.y)
        kmax = 0.0
        for i in range(nearest, len(ref.curvature)):
            k = abs(ref.curvature[i])
            if k > kmax:
                kmax = k
        return kmax

    # ---------------------------------------------------------------- track

    def _track(self, v_meas: float, v_set: float) -> Tuple[float, float]:
        err = v_set - v_meas

        # Tentative unsaturated command. Compute first to test saturation;
        # only update integrator if the command would NOT clip — classic
        # conditional integration anti-windup.
        u_unsat = self.kp * err + self.ki * self._integral
        u = max(-1.0, min(1.0, u_unsat))
        if u_unsat == u:
            self._integral += err * _DT
        # else: clipped, hold integrator

        # Deadband + split. The single-channel guarantee means the bridge
        # never sends both throttle and regen at the same time, so the
        # EMRAX motor folder (Throttle - Regen) collapses to a clean signed
        # demand without partial-cancellation surprises.
        if u >= self.deadband:
            return float(min(u, self.throttle_max)), 0.0
        if u <= -self.deadband:
            return 0.0, float(-u)
        return 0.0, 0.0


def _nearest_index(x: float, y: float, xs, ys) -> int:
    """Same helper as Pure Pursuit. Duplicated rather than shared to keep
    each strategy self-contained — when LQR/MPC arrive they may need their
    own variant (closest in trajectory time, not geometric distance)."""
    best_i = 0
    best_d2 = float("inf")
    for i, (px, py) in enumerate(zip(xs, ys)):
        dx, dy = px - x, py - y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i
