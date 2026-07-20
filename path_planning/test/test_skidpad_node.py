"""Real-node integration: PathPlanningNode delegates to the skidpad runtime.

Imports the real node (which needs fsd_path_planning), so it SKIPS in the bare
test image and RUNS in the full dv_pipeline_stack container / on the car. Proves
the delegation seam the fake-node runtime tests can't: the actual lifecycle node,
in skidpad behavior, routes to SkidpadPlannerRuntime and publishes a real
map-frame /Path carrying the figure-eight curvature — with FaSTTUBe never touched.
"""
from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("fsd_path_planning",
                    reason="full deps required (dv_pipeline_stack / car)")
rclpy = pytest.importorskip("rclpy")

from nav_msgs.msg import Odometry, Path  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402

_SK = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   os.pardir, os.pardir, "skidpad"))
if os.path.isdir(_SK):
    sys.path.insert(0, _SK)
from skidpad.skidpad_reference import SkidpadGeometry, build_reference  # noqa: E402
from path_planning.path_planning import PathPlanningNode  # noqa: E402

R = 9.125


def _odom(x, y, sp):
    m = Odometry()
    m.header.frame_id = "odom"
    m.pose.pose.position.x = float(x)
    m.pose.pose.position.y = float(y)
    m.pose.pose.orientation.w = 1.0
    m.twist.twist.linear.x = float(sp)
    return m


class _Cap(Node):
    def __init__(self):
        super().__init__("pp_skidpad_cap")
        self.paths = []
        self.finished = []
        self.create_subscription(Path, "Path", lambda m: self.paths.append(m), 50)
        self.create_subscription(
            Bool, "/slam/finished", lambda m: self.finished.append(m.data), 50)


@pytest.fixture()
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_pathplanning_node_delegates_and_publishes_figure8_path(ros):
    ex = SingleThreadedExecutor()
    cap = _Cap()
    ex.add_node(cap)
    pp = PathPlanningNode()
    pp._behavior = "skidpad"
    ex.add_node(pp)
    pp.trigger_configure()
    pp.trigger_activate()
    odom_pub = pp.create_publisher(Odometry, "/odom", 50)

    def spin(n):
        for _ in range(n):
            ex.spin_once(timeout_sec=0.005)

    # the real node routed skidpad away from FaSTTUBe
    assert pp._skidpad is not None

    ref = build_reference(SkidpadGeometry())
    spin(10)
    # drive the entry + into the first (right) circle — enough for the CW arc
    s = 0.0
    while s < 30.0:
        p = ref.sample_at(s)
        odom_pub.publish(_odom(p.x, p.y, 2.0))
        spin(8)
        s += 1.0
    spin(20)

    assert cap.paths, "real node published no /Path in skidpad mode"
    assert cap.paths[-1].header.frame_id == "map"
    assert max(len(m.poses) for m in cap.paths) >= 2
    kappas = [ps.pose.position.z for m in cap.paths for ps in m.poses]
    assert any(abs(k + 1.0 / R) < 1e-2 for k in kappas), \
        "right-circle curvature -1/R absent from the real /Path"
    # latched default-false went out on the first tick
    assert False in cap.finished

    pp.trigger_deactivate()
    pp.trigger_cleanup()
    pp.destroy_node()
    cap.destroy_node()
