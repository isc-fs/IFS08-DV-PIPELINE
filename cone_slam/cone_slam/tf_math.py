"""Pure-Python TF math helpers — no rclpy import.

Pulled out of cone_graph_slam_node so unit tests can run without
standing up the whole ROS 2 stack. Used by slam_node's
_publish_map_to_odom (Phase 2 — #382).
"""

from __future__ import annotations

import numpy as np


def compute_map_to_odom(
    slam_x: float, slam_y: float, slam_yaw: float,
    sup_x:  float, sup_y:  float, sup_yaw:  float,
) -> tuple[float, float, float]:
    """Compute the SE(2) `map → odom` transform from SLAM's absolute
    pose (map → base_link) and the supervisor's dead-reckoning
    (odom → base_link), both expressed at the same timestamp.

    The relationship comes from:
        T_map_base = T_map_odom · T_odom_base
        ⇒ T_map_odom = T_map_base · T_odom_base⁻¹

    For SE(2) that's:
        Δyaw = slam_yaw - sup_yaw
        Δpos = slam_pos - R(Δyaw) · sup_pos

    Returns (dx, dy, dyaw), with dyaw wrapped to (-π, π].
    """
    dyaw = slam_yaw - sup_yaw
    if dyaw > np.pi:
        dyaw -= 2 * np.pi
    elif dyaw < -np.pi:
        dyaw += 2 * np.pi
    c, s = np.cos(dyaw), np.sin(dyaw)
    dx = slam_x - (c * sup_x - s * sup_y)
    dy = slam_y - (s * sup_x + c * sup_y)
    return float(dx), float(dy), float(dyaw)
