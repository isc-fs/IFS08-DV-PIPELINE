"""Kinematic bicycle model.

Used today by:
  - Pure Pursuit: front-axle projection (state.xy + L·(cos yaw, sin yaw))
  - Longitudinal feedforward: lateral-grip-limited speed v = sqrt(R · a_lat_max)
    where R = 1/|kappa| from the path's curvature

Tomorrow by MPC as the plant model — same parameters, plus a discrete step()
method that integrates state forward by dt.

IFS-08 spec values (see settings.json + EmraxMotor.h):
  L_wheelbase = 1.627 m
  Front weight fraction at rest = 0.438 (rear-biased per CoG offset)
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class KinematicBicycle:
    wheelbase: float = 1.627      # IFS-08 L (m)
    max_steer_rad: float = math.radians(28.0)  # FSDSWheelFront MaxSteerAngle

    def front_axle_xy(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        return (x + self.wheelbase * math.cos(yaw),
                y + self.wheelbase * math.sin(yaw))

    def steer_for_curvature(self, kappa: float) -> float:
        """Steer angle to drive a given path curvature: tan(δ) = L · κ."""
        return math.atan(self.wheelbase * kappa)

    def normalize_steer(self, steer_rad: float) -> float:
        """Convert steering angle to normalized [-1, 1] (matches Chaos's
        SetSteeringInput convention)."""
        return max(-1.0, min(1.0, steer_rad / self.max_steer_rad))
