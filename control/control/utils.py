"""Utility functions and classes for control module.

This module provides helper functions for vector operations, angle wrapping,
QoS configuration, and derivative computation used throughout the control node.
"""

import time
from math import atan2, pi
from typing import Sequence

import numpy as np
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

# QoS profile for best-effort, low-latency message delivery
QOS_LATEST = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length (magnitude = 1).

    Args:
        vector: Input vector as numpy array

    Returns:
        Normalized vector in same direction with magnitude 1

    Raises:
        ValueError: If vector is zero-magnitude (cannot normalize)
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize zero-magnitude vector")
    return vector / norm


def millis() -> int:
    """Return monotonic time in milliseconds since arbitrary reference point.

    Uses perf_counter() which is immune to system clock adjustments.
    Rounds to nearest millisecond for compatibility with derivative calculations.

    Returns:
        Current time in milliseconds as integer
    """
    return int(round(time.perf_counter() * 1000))


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to the range [-π, π] radians.

    Converts any angle to the canonical range by adding/subtracting 2π as needed.
    Useful for comparing heading errors and avoiding angle discontinuities.

    Args:
        angle: Input angle in radians (unbounded)

    Returns:
        Equivalent angle in range [-π, π]
    """
    return (angle + pi) % (2 * pi) - pi


def angle(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Calculate the bearing angle from point p1 to point p2.

    Computes the arctangent of the displacement vector to find the heading.
    Returns 0 radians pointing East, π/2 pointing North, etc.

    Args:
        p1: Starting point [x, y]
        p2: Ending point [x, y]

    Returns:
        Bearing angle from p1 to p2 in radians, in range [-π, π]
    """
    x_disp = p2[0] - p1[0]
    y_disp = p2[1] - p1[1]
    return atan2(y_disp, x_disp)


class Derivative:
    """Numerical derivative (rate-of-change) calculator using finite differences.

    Tracks the previous value and timestamp to compute the rate of change
    via backward finite difference: (v_new - v_old) / (t_new - t_old).

    Attributes:
        v_ant: Previous value (used for finite difference)
        t_ant: Previous timestamp in milliseconds
    """

    def __init__(self) -> None:
        """Initialize the derivative calculator with zero state."""
        self.v_ant: float = 0.0
        self.t_ant: int = millis()

    def cal(self, v_nuevo: float) -> float:
        """Calculate the rate of change of the input signal.

        Computes derivative as (current_value - previous_value) / time_elapsed,
        handling edge cases like rapid repeated calls (dt ≈ 0).

        Args:
            v_nuevo: New value to compute derivative for

        Returns:
            Rate of change in units/second. Returns 0.0 if called too rapidly
            (dt < 1 millisecond) to avoid division by near-zero.
        """
        t_now = millis()
        dt_ms = t_now - self.t_ant

        # Avoid division by zero or near-zero on rapid successive calls
        if dt_ms < 1:
            return 0.0

        # Compute derivative: change in value per unit time (converted to seconds)
        derivative = ((v_nuevo - self.v_ant) / dt_ms) * 1000.0

        # Update state for next call
        self.t_ant = t_now
        self.v_ant = v_nuevo

        return derivative
