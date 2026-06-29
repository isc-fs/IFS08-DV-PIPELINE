"""Pure actuation-command scaling for the car_supervisor.

Converts the normalised control output (RuntimeControl feedback, both
in [-1, 1]) into the physical units the uDV actuation topics expect. No
ROS imports — unit-tested directly.

  * Steering. mission_control RuntimeControl.Feedback carries `steering`
    in [-1, 1] (left positive). The uDV's /steering/cmd is a Float32 in
    DEGREES (verified IFS08-DV-uDV: `steering_cmd_callback` →
    Can::sendSteeringAngle, bench P1 step 6). So steering_norm is scaled
    by a per-car `max_steering_deg`, then hard-clamped to a safety limit
    that stays well under STEERING's 70° emergency cutoff.

  * Throttle. Feedback `throttle` is [-1, 1] (negative = regen). It is
    clamped and forwarded; the real uDV throttle/brake actuation sink
    does not exist yet (firmware gap — see docs/CAR_ADAPTATION.md), so
    the supervisor publishes it on a placeholder topic for now.
"""
from __future__ import annotations

import math


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def steering_norm_to_deg(
    steering_norm: float,
    *,
    max_steering_deg: float,
    safety_limit_deg: float,
) -> float:
    """Scale steering [-1, 1] to a degrees command for /steering/cmd.

    The normalised input is first clamped to [-1, 1] (defends against a
    controller overshoot), scaled by max_steering_deg, then hard-clamped
    to ±safety_limit_deg. The safety clamp is defence-in-depth: even a
    mis-set max_steering_deg can never command past the safety limit,
    which itself must stay below STEERING's emergency cutoff (70°).

    Args:
        steering_norm: control output in [-1, 1] (left positive).
        max_steering_deg: degrees commanded at full lock (|input| == 1).
        safety_limit_deg: absolute hard ceiling on the output magnitude.

    Raises:
        ValueError: on non-finite inputs or non-positive limits.
    """
    for name, val in (
        ("steering_norm", steering_norm),
        ("max_steering_deg", max_steering_deg),
        ("safety_limit_deg", safety_limit_deg),
    ):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")
    if max_steering_deg <= 0.0:
        raise ValueError(
            f"max_steering_deg must be positive, got {max_steering_deg!r}")
    if safety_limit_deg <= 0.0:
        raise ValueError(
            f"safety_limit_deg must be positive, got {safety_limit_deg!r}")

    scaled = _clamp(steering_norm, -1.0, 1.0) * max_steering_deg
    return _clamp(scaled, -safety_limit_deg, safety_limit_deg)


def throttle_norm_clamp(throttle_norm: float) -> float:
    """Clamp throttle to [-1, 1] (negative = regen). Non-finite → 0.0.

    A non-finite throttle is the safe-state command (coast), not an
    error to raise — the relay loop must never crash on a glitchy frame.
    """
    if not math.isfinite(throttle_norm):
        return 0.0
    return _clamp(throttle_norm, -1.0, 1.0)


def safe_stop_steering_deg() -> float:
    """Steering command for the non-driving phases: centred wheel."""
    return 0.0
