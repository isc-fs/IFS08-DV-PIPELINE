"""Unified actuation output — steering + longitudinal channels from one controller tick."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActuationCommand:
    """Normalized commands after controller logic, before node boundary (sign flip, slew).

    steering_normalized: [-1, 1], positive = left (same convention as LateralController).
    throttle / regen: [0, 1]; PIVelocity emits at most one non-zero per tick. A joint
    controller (LQR, MPC) may output a coherent pair the node still clamps and slews.
    """

    steering_normalized: float = 0.0
    throttle: float = 0.0
    regen: float = 0.0
