"""Node-level behaviour tests for pipeline_watchdog_node, without ROS.

WHY A FAKE-ROS HARNESS
----------------------
The pure core (test_health_monitor) proves the *decision* logic. It cannot
prove the node is **wired** to that logic: that /dv/status actually arms it,
that the odom callback feeds speed rather than pose, that a trip actually
reaches the publisher, that the latch is not published twice. Those are the
bugs that make a watchdog silently do nothing — and "silently does nothing" is
the one failure mode a watchdog must never have.

There is no ROS on a dev laptop and no CI in this repo, so the rclpy-gated
contract tests skip on every machine that isn't the DVPC — i.e. the node's
wiring would ship unexercised. This harness stubs the *narrow* rclpy surface
the node touches (Node, QoS, the three msg types) and drives the real
PipelineWatchdogNode class through real scenarios on an injected clock.

It is deliberately NOT a substitute for a bench run: it cannot catch DDS QoS
mismatches, which is exactly why the /dv/status QoS is pinned by comment and
by test_watchdog_contract instead. It catches wiring, not transport.

ISOLATION — read before touching the stubs
------------------------------------------
The stubs are installed into sys.modules only for as long as it takes to exec
the node module, then removed again. This is not fussiness: an earlier version
left them installed, and in an environment where ROS *is* present (the humble
container) the contract test then imported the already-stubbed module from
sys.modules and silently asserted against the FAKE node instead of the real
one. A test that quietly tests a stub is worse than no test. The loaded module
keeps working after the restore because its class bases and globals are bound
at exec time; sys.modules is left exactly as we found it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

_HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REPO, "mission_control"))

_STUBBED = (
    "rclpy", "rclpy.node", "rclpy.qos",
    "std_msgs", "std_msgs.msg",
    "nav_msgs", "nav_msgs.msg",
    "fs_msgs", "fs_msgs.msg",
)


# --------------------------------------------------------------- stubs

class _FakeClock:
    """Injected time. The node reads seconds via nanoseconds * 1e-9."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self):
        return types.SimpleNamespace(nanoseconds=self.t * 1e9)


class _FakePub:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


class _FakeLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, m):
        self.lines.append(f"INFO {m}")

    def warn(self, m):
        self.lines.append(f"WARN {m}")

    def error(self, m):
        self.lines.append(f"ERROR {m}")


class _FakeNode:
    """Minimal stand-in for rclpy.node.Node — only what the watchdog uses."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._params: dict = {}
        self._clock = _FakeClock()
        self._logger = _FakeLogger()
        self.subs: dict = {}
        self.timers: list = []
        self.pubs: dict = {}

    def declare_parameter(self, name, default):
        self._params[name] = default

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self._params[name])

    def create_publisher(self, msg_type, topic, qos):
        pub = _FakePub()
        self.pubs[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic, cb, qos, **kw):
        self.subs[topic] = cb
        return types.SimpleNamespace(topic=topic)

    def create_timer(self, period, cb):
        self.timers.append((period, cb))
        return types.SimpleNamespace()

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return self._clock


def _install_stubs() -> None:
    """Install the narrow rclpy/msg surface the node imports.

    Callers MUST pair this with a restore (see _load_node_module) — leaving
    these in sys.modules makes every later test in the session import the fake.
    """
    rclpy = types.ModuleType("rclpy")
    rclpy._is_watchdog_stub = True
    rclpy.init = lambda **kw: None
    rclpy.shutdown = lambda: None
    rclpy.ok = lambda: True
    rclpy.spin = lambda n: None

    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _FakeNode
    rclpy.node = node_mod

    qos_mod = types.ModuleType("rclpy.qos")

    class _QoSProfile:
        def __init__(self, **kw):
            self.kw = kw

    qos_mod.QoSProfile = _QoSProfile
    qos_mod.ReliabilityPolicy = types.SimpleNamespace(
        RELIABLE="RELIABLE", BEST_EFFORT="BEST_EFFORT")
    qos_mod.DurabilityPolicy = types.SimpleNamespace(
        TRANSIENT_LOCAL="TRANSIENT_LOCAL", VOLATILE="VOLATILE")
    rclpy.qos = qos_mod

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = node_mod
    sys.modules["rclpy.qos"] = qos_mod

    def _msg_module(name: str, **types_):
        m = types.ModuleType(name)
        for k, v in types_.items():
            setattr(m, k, v)
        return m

    class _Bool:
        def __init__(self, data=False):
            self.data = data

    class _UInt8:
        def __init__(self, data=0):
            self.data = data

    class _Odometry:
        def __init__(self):
            pos = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            lin = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.pose = types.SimpleNamespace(
                pose=types.SimpleNamespace(position=pos))
            self.twist = types.SimpleNamespace(
                twist=types.SimpleNamespace(linear=lin))

    class _ControlCommand:
        def __init__(self):
            self.throttle = 0.0
            self.steering = 0.0
            self.brake = 0.0

    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = _msg_module("std_msgs.msg", Bool=_Bool, UInt8=_UInt8)
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs.msg

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs.msg = _msg_module("nav_msgs.msg", Odometry=_Odometry)
    sys.modules["nav_msgs"] = nav_msgs
    sys.modules["nav_msgs.msg"] = nav_msgs.msg

    fs_msgs = types.ModuleType("fs_msgs")
    fs_msgs.msg = _msg_module("fs_msgs.msg", ControlCommand=_ControlCommand)
    sys.modules["fs_msgs"] = fs_msgs
    sys.modules["fs_msgs.msg"] = fs_msgs.msg


def _load_node_module():
    """Exec the real node module against the stubs, then restore sys.modules.

    The returned module keeps working — its class bases and globals are bound
    during exec_module — while the rest of the pytest session sees the genuine
    rclpy (or no rclpy) exactly as it did before.
    """
    saved = {name: sys.modules.get(name) for name in _STUBBED}
    _install_stubs()
    try:
        spec = importlib.util.spec_from_file_location(
            "_pipeline_watchdog_node_stubbed",
            os.path.join(ROOT, "pipeline_watchdog", "pipeline_watchdog_node.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        # Never leave the stubbed node module discoverable under the real name.
        sys.modules.pop("_pipeline_watchdog_node_stubbed", None)


_node_mod = _load_node_module()
PipelineWatchdogNode = _node_mod.PipelineWatchdogNode
# Take the message classes from the loaded module's own globals — they are the
# stub types it was exec'd against, whatever sys.modules holds now.
Bool = _node_mod.Bool
UInt8 = _node_mod.UInt8
Odometry = _node_mod.Odometry
ControlCommand = _node_mod.ControlCommand

from mission_control.interface_contract import (  # noqa: E402
    DV_IDLE,
    DV_RUNNING,
    TOPIC_WATCHDOG_EMERGENCY,
)


# --------------------------------------------------------------- helpers

class Harness:
    def __init__(self) -> None:
        self.node = PipelineWatchdogNode()
        self.pub = self.node.pubs[TOPIC_WATCHDOG_EMERGENCY]
        _, self.tick = self.node.timers[0]

    @property
    def t(self) -> float:
        return self.node._clock.t

    def at(self, t: float) -> "Harness":
        self.node._clock.t = t
        return self

    def dv_status(self, byte: int) -> None:
        self.node.subs["/dv/status"](UInt8(data=byte))

    def pose(self, x=0.0, y=0.0) -> None:
        m = Odometry()
        m.pose.pose.position.x = x
        m.pose.pose.position.y = y
        self.node.subs["/slam/pose"](m)

    def odom(self, vx=0.0) -> None:
        m = Odometry()
        m.twist.twist.linear.x = vx
        self.node.subs["/odom"](m)

    def cmd(self) -> None:
        self.node.subs["/ctrl/cmd_internal"](ControlCommand())

    def feed_all(self, x=0.0, vx=0.0) -> None:
        self.pose(x=x)
        self.odom(vx=vx)
        self.cmd()

    @property
    def raised(self) -> bool:
        return any(m.data for m in self.pub.published)

    def run(self) -> None:
        self.tick()


@pytest.fixture
def h():
    return Harness()


# ------------------------------------------------------------- isolation

def test_stubs_do_not_leak_into_sys_modules():
    """Regression guard. An earlier version left the stubs installed, and in
    the ROS container the contract test then imported the stubbed module from
    sys.modules and asserted against the FAKE node while looking green.

    If rclpy is real here, it must still be real after this module imported.
    If rclpy is absent here, it must still be absent — not silently faked.
    """
    rclpy_now = sys.modules.get("rclpy")
    assert not getattr(rclpy_now, "_is_watchdog_stub", False), \
        "fake rclpy leaked into sys.modules — other tests will import the stub"
    assert "_pipeline_watchdog_node_stubbed" not in sys.modules


def test_harness_drives_the_real_node_class():
    """The harness must exercise the genuine node source, not a copy of it."""
    src = os.path.join(ROOT, "pipeline_watchdog", "pipeline_watchdog_node.py")
    assert os.path.samefile(_node_mod.__file__, src)


# ---------------------------------------------------------------- wiring

def test_subscribes_to_exactly_the_expected_topics(h):
    assert set(h.node.subs) == {
        "/dv/status", "/slam/pose", "/odom", "/ctrl/cmd_internal"}


def test_publishes_default_false_on_startup(h):
    """A late-joining mission_control must see a defined state."""
    assert len(h.pub.published) == 1
    assert h.pub.published[0].data is False


def test_ticks_at_a_sane_rate(h):
    period, _ = h.node.timers[0]
    assert 0.0 < period <= 0.1


# --------------------------------------------------------------- arming

def test_does_not_arm_until_dv_running(h):
    h.at(0.0).dv_status(DV_IDLE)
    h.at(100.0).run()          # long past any grace window
    assert not h.raised, "watchdog tripped while the pipeline was not running"


def test_arms_on_dv_running(h):
    h.at(0.0).dv_status(DV_RUNNING)
    assert h.node._monitor.armed


def test_disarm_on_leaving_running_republishes_false(h):
    """A new run must start from a defined false, not a stale latched true."""
    h.at(0.0).dv_status(DV_RUNNING)
    h.at(100.0).run()
    assert h.raised
    h.at(101.0).dv_status(DV_IDLE)
    assert h.pub.published[-1].data is False
    assert not h.node._monitor.armed


# ------------------------------------------------------- the real thing

def test_healthy_running_pipeline_never_trips(h):
    """The false-positive case. A false trip fires the EBS at speed."""
    h.at(0.0).dv_status(DV_RUNNING)
    t, x = 0.0, 0.0
    while t < 60.0:
        x += 3.0 * 0.05
        h.at(t).feed_all(x=x, vx=3.0)
        h.run()
        assert not h.raised, f"false trip at t={t}"
        t += 0.05


def test_slam_pose_freeze_trips_and_requests_emergency(h):
    """The documented runaway: pose stops, control keeps driving on cache."""
    h.at(0.0).dv_status(DV_RUNNING)
    t = 0.0
    while t < 10.0:                      # healthy, past the grace window
        h.at(t).feed_all(x=3.0 * t, vx=3.0)
        h.run()
        t += 0.05
    assert not h.raised
    # SLAM dies. Everything else keeps running — exactly the sick-not-dead case.
    while t < 11.0:
        h.at(t).odom(vx=3.0)
        h.at(t).cmd()
        h.run()
        t += 0.05
    assert h.raised, "watchdog missed a frozen /slam/pose"
    assert h.pub.published[-1].data is True
    assert any("WATCHDOG TRIPPED" in ln for ln in h.node._logger.lines)
    assert any("/slam/pose" in ln for ln in h.node._logger.lines)


def test_emergency_is_published_once_not_every_tick(h):
    """The latch must not spam the channel at 20 Hz."""
    h.at(0.0).dv_status(DV_RUNNING)
    h.at(100.0)
    for _ in range(50):
        h.run()
    trues = [m for m in h.pub.published if m.data]
    assert len(trues) == 1


def test_odom_feeds_speed_not_pose(h):
    """Wiring guard: if /odom were wired to record_pose (or /slam/pose to
    record_speed), the frozen-pose check would read its speed from the very
    source it is meant to be auditing, and never trip."""
    h.at(0.0).dv_status(DV_RUNNING)
    t = 0.0
    # Pose is frozen at the origin the whole time; only /odom reports motion.
    while t < 12.0:
        h.at(t).feed_all(x=0.0, vx=5.0)
        h.run()
        t += 0.05
    assert h.raised
    assert any("not advancing" in ln for ln in h.node._logger.lines)


def test_stopped_car_with_frozen_pose_never_trips(h):
    """A car legitimately stopped at the finish line has a frozen pose."""
    h.at(0.0).dv_status(DV_RUNNING)
    t = 0.0
    while t < 60.0:
        h.at(t).feed_all(x=0.0, vx=0.0)
        h.run()
        assert not h.raised, f"false trip on a stopped car at t={t}"
        t += 0.05


def test_grace_window_covers_slow_startup(h):
    """Nodes JIT-warming for a few seconds must not trip the watchdog."""
    h.at(0.0).dv_status(DV_RUNNING)
    t = 0.0
    while t < 4.0:            # under the 5 s default grace, nothing publishes
        h.at(t).run()
        assert not h.raised, f"tripped during spin-up at t={t}"
        t += 0.05
