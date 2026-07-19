"""Tests for SkidpadPlannerRuntime against a fake ROS node.

Needs the real message types (nav_msgs/std_msgs), so it runs in the ROS
container, not the host venv. It drives the runtime through its lifecycle and a
simulated /odom stream and asserts the wire contract path_planning_node and
mission_control depend on: a curvature-carrying /Path window on the odom-frame
poses, and a single latched /slam/finished rising edge at end + standstill.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

pytest.importorskip("nav_msgs", reason="ROS message types required (run in container)")

from nav_msgs.msg import Odometry, Path  # noqa: E402

from skidpad.skidpad_planner_runtime import SkidpadPlannerRuntime  # noqa: E402
from skidpad.skidpad_reference import SkidpadGeometry, build_reference  # noqa: E402

R = 9.125


# ---------------------------------------------------------------- fakes

class _Param:
    def __init__(self, value):
        self.value = value


class _Logger:
    def info(self, *a):
        pass

    def warn(self, *a):
        pass

    def error(self, *a):
        pass


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Sub:
    pass


class FakeNode:
    """Just enough of the rclpy node surface for SkidpadPlannerRuntime."""

    def __init__(self):
        self._params: dict = {}
        self.pubs: dict = {}
        self.sub_cb = None
        self._sub = None

    # params
    def declare_parameter(self, name, value):
        self._params[name] = _Param(value)
        return self._params[name]

    def has_parameter(self, name):
        return name in self._params

    def get_parameter(self, name):
        return self._params[name]

    # pub/sub
    def create_lifecycle_publisher(self, msg_type, topic, qos):
        p = _Pub()
        self.pubs[topic] = p
        return p

    def create_subscription(self, msg_type, topic, cb, qos):
        self.sub_cb = cb
        self._sub = _Sub()
        return self._sub

    def destroy_subscription(self, sub):
        self.sub_cb = None
        self._sub = None

    def destroy_publisher(self, pub):
        pass

    def get_logger(self):
        return _Logger()


def _odom(x, y, speed):
    m = Odometry()
    m.pose.pose.position.x = float(x)
    m.pose.pose.position.y = float(y)
    m.twist.twist.linear.x = float(speed)
    return m


def _configured_runtime(**params):
    node = FakeNode()
    rt = SkidpadPlannerRuntime(node)
    rt.on_configure()
    for k, v in params.items():
        node.get_parameter(k).value = v
    return node, rt


# ------------------------------------------------------------- lifecycle

def test_configure_creates_path_and_finished_publishers():
    node, rt = _configured_runtime()
    assert "Path" in node.pubs
    assert "/slam/finished" in node.pubs


def test_activate_subscribes_and_latches_finished_false_on_first_odom():
    node, rt = _configured_runtime()
    rt.on_activate()
    assert node.sub_cb is not None                  # subscribed to /odom
    assert not node.pubs["/slam/finished"].msgs     # nothing until first tick
    node.sub_cb(_odom(0.0, 0.0, 0.0))               # first odom
    fin = node.pubs["/slam/finished"].msgs
    assert fin and fin[0].data is False             # latched default now sent


def test_deactivate_then_cleanup_drops_io():
    node, rt = _configured_runtime()
    rt.on_activate()
    rt.on_deactivate()
    assert node.sub_cb is None
    rt.on_cleanup()  # must not raise


# ------------------------------------------------------------ /Path output

def test_publishes_path_on_odom_with_curvature_sidechannel():
    node, rt = _configured_runtime()
    rt.on_activate()
    # a pose on the first right circle → curvature must be −1/R, encoded in z
    node.sub_cb(_odom(0.0, 0.0, 1.0))       # anchor at spawn
    g = SkidpadGeometry()
    # drive to ~quarter into the first right lap
    s = g.entry_len_m + 0.5 * math.pi * R
    p = build_reference(g).sample_at(s)
    node.sub_cb(_odom(p.x, p.y, 1.0))
    paths = node.pubs["Path"].msgs
    assert paths, "expected a /Path publish"
    last = paths[-1]
    assert isinstance(last, Path)
    assert len(last.poses) >= 2
    # curvature side-channel present on the circle portion
    kappas = [ps.pose.position.z for ps in last.poses]
    assert any(abs(k + 1.0 / R) < 1e-3 for k in kappas), \
        "right-circle curvature -1/R should appear in pose.position.z"


def test_path_frame_is_the_configured_frame():
    node, rt = _configured_runtime()
    rt.on_activate()
    node.sub_cb(_odom(0.0, 0.0, 1.0))
    node.sub_cb(_odom(2.0, 0.0, 1.0))
    assert node.pubs["Path"].msgs[-1].header.frame_id == "map"


# --------------------------------------------------------- /slam/finished

def _drive_full_run(node, geom):
    ref = build_reference(geom)
    node.sub_cb(_odom(0.0, 0.0, 2.0))
    s = 0.0
    while s < ref.total_length:
        p = ref.sample_at(s)
        node.sub_cb(_odom(p.x, p.y, 2.0))
        s += 0.5
    end = ref.sample_at(ref.total_length)
    for _ in range(3):
        node.sub_cb(_odom(end.x, end.y, 0.0))


def test_finished_rises_once_at_end_and_standstill():
    node, rt = _configured_runtime()
    rt.on_activate()
    _drive_full_run(node, SkidpadGeometry())
    trues = [m for m in node.pubs["/slam/finished"].msgs if m.data is True]
    assert len(trues) == 1, "exactly one rising /slam/finished=true"


def test_not_finished_while_moving_at_end():
    node, rt = _configured_runtime()
    rt.on_activate()
    ref = build_reference(SkidpadGeometry())
    node.sub_cb(_odom(0.0, 0.0, 2.0))
    s = 0.0
    while s < ref.total_length:
        p = ref.sample_at(s)
        node.sub_cb(_odom(p.x, p.y, 2.0))   # never stops
        s += 0.5
    trues = [m for m in node.pubs["/slam/finished"].msgs if m.data is True]
    assert not trues, "must not finish while still moving"


def test_window_empties_and_path_goes_zero_length_at_end():
    node, rt = _configured_runtime()
    rt.on_activate()
    _drive_full_run(node, SkidpadGeometry())
    assert len(node.pubs["Path"].msgs[-1].poses) == 0
