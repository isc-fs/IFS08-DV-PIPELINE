"""Vehicle state contract — shared by all controllers (lateral, longitudinal,
future LQR/MPC).

Single source of truth for "what the controller knows about the car this tick".
Populated once per tick by the ROS node from /cone_slam/state Odometry, then
passed by reference into every controller.compute() call.

Conventions (ISO 8855, matches cone_slam frame):
  - x forward, y left, z up
  - yaw measured CCW from +x
  - vx is longitudinal (along car heading); vy lateral (left positive)
  - all units SI (m, m/s, rad, rad/s)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleState:
    # Pose in odom frame
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    # Body-frame velocity (twist in child_frame_id from Odometry)
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    # Optional: body-frame acceleration if/when we wire IMU through
    ax: float = 0.0
    ay: float = 0.0

    @property
    def speed(self) -> float:
        """Scalar speed (always non-negative)."""
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5
