"""Reference trajectory contract — what the controllers track.

A path published by Plan_Path is a sequence of (x, y) waypoints in the odom
frame. ReferenceTrajectory wraps that with the derived geometry every
controller needs (yaw, curvature, cumulative arc length) plus a per-point
speed profile filled in by the longitudinal controller's setpoint stage.

The same object is passed to lateral.compute() and longitudinal.compute() each
tick — keep it lean, no per-controller state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import List


@dataclass
class ReferenceTrajectory:
    # Geometry — populated from /Path
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)         # tangent heading at each pt
    curvature: List[float] = field(default_factory=list)   # 1/m, signed
    s: List[float] = field(default_factory=list)           # cumulative arc length

    # Speed profile — populated by longitudinal pre-pass before compute()
    v_ref: List[float] = field(default_factory=list)       # m/s per point

    # Stop semantics — set by stop-detection logic
    stop_distance: float = float("inf")  # arc-length distance from car to stop
    stop_latched: bool = False

    @property
    def empty(self) -> bool:
        return len(self.x) < 2

    @property
    def length(self) -> float:
        return self.s[-1] if self.s else 0.0

    @classmethod
    def from_xy(cls, xs: List[float], ys: List[float]) -> "ReferenceTrajectory":
        """Build from raw waypoints — derives yaw, curvature, arc length."""
        if len(xs) != len(ys) or len(xs) < 2:
            return cls(x=list(xs), y=list(ys))

        n = len(xs)
        yaws = [0.0] * n
        s = [0.0] * n
        # Tangent heading + arc length (forward differences, last point copies prev)
        for i in range(n - 1):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            yaws[i] = math.atan2(dy, dx)
            s[i + 1] = s[i] + math.hypot(dx, dy)
        yaws[-1] = yaws[-2]

        # Curvature via discrete heading derivative wrt arc length
        kappa = [0.0] * n
        for i in range(1, n - 1):
            ds = s[i + 1] - s[i - 1]
            if ds > 1e-6:
                dyaw = _wrap_pi(yaws[i + 1] - yaws[i - 1])
                kappa[i] = dyaw / ds
        kappa[0] = kappa[1] if n > 1 else 0.0
        kappa[-1] = kappa[-2] if n > 1 else 0.0

        return cls(x=list(xs), y=list(ys), yaw=yaws, curvature=kappa, s=s)


def _wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
