"""Real-node integration: cone_slam skidpad behavior is a pose passthrough.

Imports the real SLAM node (needs gtsam), so it SKIPS in the bare test image and
RUNS in the full dv_pipeline_stack container / on the car. Proves that in skidpad
behavior the graph is disabled and /slam/pose is the EKF /odom pose republished
in the map frame — the wire behavior the pure passthrough builders imply.
"""
from __future__ import annotations

import pytest

pytest.importorskip("gtsam", reason="full deps required (dv_pipeline_stack / car)")
rclpy = pytest.importorskip("rclpy")

from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from cone_slam.cone_graph_slam_node import ConeGraphSlamNode  # noqa: E402


class _Cap(Node):
    def __init__(self):
        super().__init__("slam_skidpad_cap")
        self.poses = []
        self.create_subscription(
            Odometry, "/slam/pose", lambda m: self.poses.append(m), 50)


@pytest.fixture()
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_slam_skidpad_republishes_odom_as_slam_pose(ros):
    ex = SingleThreadedExecutor()
    cap = _Cap()
    ex.add_node(cap)
    cs = ConeGraphSlamNode()
    cs._behavior = "skidpad"
    ex.add_node(cs)
    cs.trigger_configure()
    cs.trigger_activate()
    odom_pub = cs.create_publisher(Odometry, "/odom", 50)

    def spin(n):
        for _ in range(n):
            ex.spin_once(timeout_sec=0.005)

    assert cs._skidpad_passthrough is True    # graph disabled

    spin(10)
    m = Odometry()
    m.header.frame_id = "odom"
    m.pose.pose.position.x = 3.5
    m.pose.pose.position.y = -1.25
    m.pose.pose.orientation.w = 0.92388
    m.pose.pose.orientation.z = 0.38268      # ~45°
    m.twist.twist.linear.x = 2.0
    for _ in range(5):
        odom_pub.publish(m)
        spin(8)

    assert cap.poses, "no /slam/pose from the passthrough"
    last = cap.poses[-1]
    assert last.header.frame_id == "map"                 # relabelled to map
    assert last.pose.pose.position.x == pytest.approx(3.5, abs=1e-6)
    assert last.pose.pose.position.y == pytest.approx(-1.25, abs=1e-6)
    assert last.pose.pose.orientation.z == pytest.approx(0.38268, abs=1e-5)
    assert last.twist.twist.linear.x == pytest.approx(2.0, abs=1e-6)

    cs.trigger_deactivate()
    cs.trigger_cleanup()
    cs.destroy_node()
    cap.destroy_node()
