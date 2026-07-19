"""End-to-end DDS integration test for the deterministic skidpad planner.

Runs the real SkidpadPlannerRuntime on a real rclpy LifecycleNode, driven over
the wire by a /odom publisher, with /Path and /slam/finished captured through
real DDS. This is what the fake-node runtime tests cannot cover: lifecycle
publisher activation, latched (TRANSIENT_LOCAL) QoS delivery to a late joiner,
and actual message transport. Runs only in the ROS container.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

rclpy = pytest.importorskip("rclpy", reason="rclpy required (run in container)")

from nav_msgs.msg import Odometry, Path  # noqa: E402
from rclpy.lifecycle import LifecycleNode  # noqa: E402
from rclpy.lifecycle import TransitionCallbackReturn  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool  # noqa: E402

from skidpad.skidpad_planner_runtime import SkidpadPlannerRuntime  # noqa: E402
from skidpad.skidpad_reference import SkidpadGeometry, build_reference  # noqa: E402

R = 9.125

_LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class _PlannerNode(LifecycleNode):
    """Real lifecycle node delegating to the runtime, exactly as
    path_planning_node does in skidpad mode."""

    def __init__(self, overrides=None):
        super().__init__("skidpad_itest_planner",
                         parameter_overrides=overrides or [])
        self.rt = SkidpadPlannerRuntime(self)

    def on_configure(self, state):
        self.rt.on_configure()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.rt.on_activate()
        return super().on_activate(state)

    def on_cleanup(self, state):
        self.rt.on_cleanup()
        return TransitionCallbackReturn.SUCCESS


class _Capture(rclpy.node.Node):
    def __init__(self):
        super().__init__("skidpad_itest_capture")
        self.paths: list = []
        self.finished: list = []
        self.create_subscription(Path, "Path", lambda m: self.paths.append(m), 10)
        self.create_subscription(
            Bool, "/slam/finished", lambda m: self.finished.append(m.data), _LATCHED)


def _odom(x, y, speed):
    m = Odometry()
    m.header.frame_id = "odom"
    m.pose.pose.position.x = float(x)
    m.pose.pose.position.y = float(y)
    m.twist.twist.linear.x = float(speed)
    return m


@pytest.fixture()
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def _spin(nodes, n=3):
    for _ in range(n):
        for nd in nodes:
            rclpy.spin_once(nd, timeout_sec=0.01)


def test_end_to_end_drive_publishes_path_and_finishes(ros):
    # Shorten the run so the DDS drive is quick but still a full figure-eight.
    P = rclpy.parameter.Parameter
    overrides = [
        P("skidpad_entry_len_m", P.Type.DOUBLE, 4.0),
        P("skidpad_exit_len_m", P.Type.DOUBLE, 6.0),
        P("skidpad_laps_per_side", P.Type.INTEGER, 1),
    ]
    planner = _PlannerNode(overrides)
    planner.trigger_configure()   # runtime reads the overridden params here
    planner.trigger_activate()

    odom_pub = planner.create_publisher(Odometry, "/odom", 10)
    cap = _Capture()
    nodes = [planner, cap]

    geom = SkidpadGeometry(entry_len_m=4.0, exit_len_m=6.0, laps_per_side=1)
    ref = build_reference(geom)
    _spin(nodes, 5)   # let discovery settle before driving

    # Drive the whole reference over DDS, then a few stopped ticks.
    s = 0.0
    while s < ref.total_length:
        p = ref.sample_at(s)
        odom_pub.publish(_odom(p.x, p.y, 2.0))
        _spin(nodes, 1)
        s += 1.0
    end = ref.sample_at(ref.total_length)
    for _ in range(6):
        odom_pub.publish(_odom(end.x, end.y, 0.0))
        _spin(nodes, 2)
    _spin(nodes, 10)

    # A /Path was delivered, in the map frame, with the curvature side-channel.
    assert cap.paths, "no /Path delivered over DDS"
    mid = next((m for m in cap.paths if len(m.poses) >= 2), None)
    assert mid is not None
    assert mid.header.frame_id == "map"
    kappas = [ps.pose.position.z for m in cap.paths for ps in m.poses]
    assert any(abs(k + 1.0 / R) < 1e-2 for k in kappas), "right-circle κ missing"

    # /slam/finished went false (start) → true (end) end-to-end over DDS.
    assert False in cap.finished, "latched default-false never delivered"
    assert cap.finished[-1] is True, "skidpad never reported finished over DDS"

    planner.trigger_deactivate()
    planner.trigger_cleanup()
    planner.destroy_node()
    cap.destroy_node()
