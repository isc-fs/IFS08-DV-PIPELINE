"""Tests for the deterministic-skidpad pose passthrough builders.

Only needs nav_msgs/geometry_msgs (no gtsam), so it runs in the ROS container.
Pins the contract: /slam/pose carries the EKF /odom pose unchanged, only
relabelled to the map frame; map→odom is identity.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

pytest.importorskip("nav_msgs", reason="ROS message types required (run in container)")

from nav_msgs.msg import Odometry  # noqa: E402

from cone_slam.skidpad_passthrough import (  # noqa: E402
    identity_map_to_odom,
    passthrough_pose,
)


def _odom():
    m = Odometry()
    m.header.frame_id = "odom"
    m.header.stamp.sec = 42
    m.pose.pose.position.x = 3.5
    m.pose.pose.position.y = -1.25
    m.pose.pose.orientation.w = 0.9238795
    m.pose.pose.orientation.z = 0.3826834      # ~45°
    m.pose.covariance = [float(i) for i in range(36)]
    m.twist.twist.linear.x = 2.0
    m.twist.twist.angular.z = 0.3
    m.twist.covariance = [float(i) * 2 for i in range(36)]
    return m


def test_pose_is_copied_verbatim_into_map_frame():
    src = _odom()
    out = passthrough_pose(src, "map", "base_link")
    assert out.header.frame_id == "map"          # relabelled
    assert out.child_frame_id == "base_link"
    # pose unchanged — the whole point: no drift correction applied
    assert out.pose.pose.position.x == src.pose.pose.position.x
    assert out.pose.pose.position.y == src.pose.pose.position.y
    assert out.pose.pose.orientation.z == src.pose.pose.orientation.z
    assert out.pose.pose.orientation.w == src.pose.pose.orientation.w


def test_twist_and_covariance_preserved():
    src = _odom()
    out = passthrough_pose(src, "map", "base_link")
    assert out.twist.twist.linear.x == 2.0
    assert out.twist.twist.angular.z == 0.3
    assert list(out.pose.covariance) == list(src.pose.covariance)
    assert list(out.twist.covariance) == list(src.twist.covariance)


def test_stamp_preserved():
    src = _odom()
    out = passthrough_pose(src, "map", "base_link")
    assert out.header.stamp.sec == 42


def test_identity_map_to_odom():
    src = _odom()
    t = identity_map_to_odom(src.header.stamp, "map", "odom")
    assert t.header.frame_id == "map"
    assert t.child_frame_id == "odom"
    assert t.transform.rotation.w == 1.0
    assert t.transform.rotation.x == 0.0
    assert t.transform.rotation.y == 0.0
    assert t.transform.rotation.z == 0.0
    assert t.transform.translation.x == 0.0
    assert t.transform.translation.y == 0.0
