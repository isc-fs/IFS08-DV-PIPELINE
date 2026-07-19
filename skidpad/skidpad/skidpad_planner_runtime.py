"""ROS runtime for the deterministic skidpad planner.

This is the thin ROS shell that `path_planning_node` delegates to when its
behavior is ``skidpad``. It owns the node's skidpad-mode I/O and hands every
decision to the pure `SkidpadDriver`, so the FaSTTUBe code path in
path_planning_node is never entered for skidpad (and is byte-identical for every
other mission — the isolation the operator asked for).

I/O (all in the odom frame; see skidpad-deterministic-design memory):

  subscribes  /odom            (nav_msgs/Odometry) — EKF dead-reckoned pose +
                                body twist. The pose we trust; no cones, no TF.
  publishes   Path             (nav_msgs/Path)     — forward window the
                                controller tracks; per-pose curvature smuggled
                                in pose.position.z (same side-channel as
                                path_planning._pose_stamped).
              /slam/finished   (std_msgs/Bool, latched) — rises once the car
                                has run the whole figure-eight AND stopped.
                                mission_control turns it into DV_FINISHED.

Why publish /slam/finished from here and not from slam_node: slam_node's skidpad
behavior is a pure pose passthrough (no graph, no progress), so the finish
detector lives with the arc-length progress — which is here. slam_node still
latches /slam/finished=false on activate; mission_control only ACTS on a rising
true and never un-sets on false, so the two publishers coexist safely.

The controller does NOT decelerate early (no perception, no speed side-channel):
the driver empties the window near the end, control fail-safes to a zero
command, and the brakeless car coasts to a stop — then /slam/finished latches.
"""
from __future__ import annotations

import math
from typing import Optional

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from skidpad.skidpad_driver import SkidpadDriver
from skidpad.skidpad_reference import SkidpadGeometry, build_reference

# /slam/finished is latched so a late-joining mission_control inherits the
# value. Mirrors slam_node's finished_qos exactly (RELIABLE + TRANSIENT_LOCAL).
_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

_TOPIC_ODOM = "/odom"
_TOPIC_PATH = "Path"
_TOPIC_FINISHED = "/slam/finished"


def _yaw_to_quat(yaw: float) -> tuple[float, float]:
    """(w, z) of the yaw-only quaternion; x=y=0 in the flat-2D world."""
    return math.cos(yaw * 0.5), math.sin(yaw * 0.5)


def build_path_msg(points, frame_id: str, stamp) -> Path:
    """Pack driver PathPoints into a nav_msgs/Path.

    Curvature rides in pose.position.z — the exact side-channel
    path_planning._pose_stamped uses and control._on_path reads back. Keeping
    the encoding identical is what lets control_node stay UNCHANGED for skidpad.
    """
    out = Path()
    out.header.frame_id = frame_id
    out.header.stamp = stamp
    for p in points:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = stamp
        ps.pose.position.x = float(p.x)
        ps.pose.position.y = float(p.y)
        ps.pose.position.z = float(p.curvature)   # side-channel, not height
        qw, qz = _yaw_to_quat(p.yaw)
        ps.pose.orientation.w = qw
        ps.pose.orientation.z = qz
        out.poses.append(ps)
    return out


class SkidpadPlannerRuntime:
    """Owns path_planning_node's skidpad-mode lifecycle + I/O. Construct once in
    the node; the node forwards its lifecycle transitions here."""

    def __init__(self, node) -> None:
        self._node = node
        self._driver: Optional[SkidpadDriver] = None
        self._path_pub = None
        self._finished_pub = None
        self._sub_odom = None
        self._finished_latched = False
        self._sent_initial_finished = False
        self._frame_id = "map"

    # ------------------------------------------------------------- params
    def declare_params(self) -> None:
        """Declare skidpad params on the node (idempotent). Kept here so all
        skidpad configuration lives in the skidpad package."""
        d = self._node.declare_parameter
        _dcl = lambda name, val: (  # noqa: E731 — terse guarded declare
            None if self._node.has_parameter(name) else d(name, val))
        _dcl("skidpad_entry_len_m", 8.0)
        _dcl("skidpad_exit_len_m", 15.0)
        _dcl("skidpad_laps_per_side", 2)
        _dcl("skidpad_window_len_m", 10.0)
        _dcl("skidpad_standstill_mps", 0.5)
        _dcl("skidpad_finish_margin_m", 0.5)
        _dcl("skidpad_path_frame", "map")

    def _geometry(self) -> SkidpadGeometry:
        g = self._node.get_parameter
        return SkidpadGeometry(
            entry_len_m=float(g("skidpad_entry_len_m").value),
            exit_len_m=float(g("skidpad_exit_len_m").value),
            laps_per_side=int(g("skidpad_laps_per_side").value),
        )

    # -------------------------------------------------------- lifecycle
    def on_configure(self) -> None:
        self.declare_params()
        g = self._node.get_parameter
        self._frame_id = str(g("skidpad_path_frame").value)
        reference = build_reference(self._geometry())
        self._driver = SkidpadDriver(
            reference,
            window_len_m=float(g("skidpad_window_len_m").value),
            standstill_mps=float(g("skidpad_standstill_mps").value),
            finish_margin_m=float(g("skidpad_finish_margin_m").value),
        )
        self._path_pub = self._node.create_lifecycle_publisher(
            Path, _TOPIC_PATH, 10)
        self._finished_pub = self._node.create_lifecycle_publisher(
            Bool, _TOPIC_FINISHED, _LATCHED_QOS)
        self._node.get_logger().info(
            "skidpad deterministic planner configured — reference length "
            f"{reference.total_length:.1f} m, window "
            f"{float(g('skidpad_window_len_m').value):.1f} m, frame "
            f"'{self._frame_id}'")

    def on_activate(self) -> None:
        if self._driver is not None:
            self._driver.reset()
        self._finished_latched = False
        # The latched default-false is published on the FIRST odom tick, not
        # here: this runs before the node's super().on_activate() flips the
        # lifecycle publishers active, and an inactive lifecycle publisher
        # silently drops the message — the latch would never reach a late
        # joiner. By the first callback the node is fully active.
        self._sent_initial_finished = False
        self._sub_odom = self._node.create_subscription(
            Odometry, _TOPIC_ODOM, self._on_odom, 10)
        self._node.get_logger().info(
            "skidpad planner active — tracking the figure-eight off /odom")

    def on_deactivate(self) -> None:
        if self._sub_odom is not None:
            self._node.destroy_subscription(self._sub_odom)
            self._sub_odom = None

    def on_cleanup(self) -> None:
        if self._sub_odom is not None:
            self._node.destroy_subscription(self._sub_odom)
            self._sub_odom = None
        for pub in (self._path_pub, self._finished_pub):
            if pub is not None:
                self._node.destroy_publisher(pub)
        self._path_pub = None
        self._finished_pub = None
        self._driver = None

    # ------------------------------------------------------------- odom
    def _on_odom(self, msg: Odometry) -> None:
        if self._driver is None or self._path_pub is None:
            return
        # Latched default-false, published once from an active publisher.
        if not self._sent_initial_finished:
            self._sent_initial_finished = True
            if self._finished_pub is not None:
                self._finished_pub.publish(Bool(data=False))
        pose_x = float(msg.pose.pose.position.x)
        pose_y = float(msg.pose.pose.position.y)
        v = msg.twist.twist.linear
        speed = math.hypot(float(v.x), float(v.y))

        out = self._driver.step(pose_x, pose_y, speed)

        # Publish the forward window (empty near the end → control coasts).
        self._path_pub.publish(
            build_path_msg(out.path, self._frame_id, msg.header.stamp))

        # Latch /slam/finished on the rising edge only.
        if out.finished and not self._finished_latched:
            self._finished_latched = True
            if self._finished_pub is not None:
                self._finished_pub.publish(Bool(data=True))
            self._node.get_logger().info(
                "skidpad complete — figure-eight driven and stopped; "
                f"{_TOPIC_FINISHED}=true (→ DV_FINISHED)")
