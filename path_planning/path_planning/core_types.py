"""Shared dataclasses for the path planning package.

These types form the input/output contract between the ROS node
(`path_planning.py`) and the planner adapter (`fasttube_adapter.py`).
Kept dependency-free (numpy only) so the node and tests can import them
without pulling in the planner library itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ConeColor(IntEnum):
    """Local mirror of cone_slam.color_classifier.ConeColor."""

    YELLOW = 0       # right side of track
    BLUE = 1         # left side of track
    ORANGE = 2       # small orange (centerline / waypoint)
    BIG_ORANGE = 3   # large orange (start/finish)


@dataclass(frozen=True)
class Cone:
    x: float          # world frame
    y: float          # world frame
    color: ConeColor


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float        # radians


@dataclass(frozen=True)
class PathPoint:
    x: float           # world frame
    y: float           # world frame
    yaw: float         # radians, tangent direction
    # Signed curvature (1/m) at this path point. Positive = left turn,
    # negative = right turn, 0 on straights. Sourced from FaSTTUBe's
    # analytical B-spline curvature when available; controller-side
    # finite-difference fallback when not.
    curvature: float = 0.0


def world_to_body(pts_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Project (N, 2) world-frame points into body frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy.copy()
    cos_y, sin_y = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    R = np.array([[cos_y, sin_y], [-sin_y, cos_y]], dtype=np.float64)
    return (pts_xy - np.array([pose.x, pose.y], dtype=np.float64)) @ R.T


def body_to_world(pts_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Project (N, 2) body-frame points into world frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy.copy()
    cos_y, sin_y = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    return pts_xy @ R.T + np.array([pose.x, pose.y], dtype=np.float64)
