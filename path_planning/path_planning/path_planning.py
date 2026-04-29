"""Path planning ROS 2 node — thin adapter around path_planning.planner.

Subscribes:
  /Conos          (visualization_msgs/MarkerArray) — world-frame cone map
                  from cone_graph_slam. Each marker carries:
                    - pose.position (x, y, z)        — world-frame cone center
                    - color (r, g, b)                — encoded ConeColor
                    - id                             — persistent landmark id

Publishes:
  /Path           (nav_msgs/Path) — interpolated centerline in `odom`,
                  consumed by control.

Looks up TF:
  odom → base_link  for car position + heading. cone_graph_slam owns
                    this transform; if it's missing we skip the tick
                    rather than fall back on a stale or wrong frame.

Algorithm: delegated to `planner.plan_centerline` (forward-walking
midpoint with cubic-spline smoothing). This module is only the ROS 2
plumbing — color decoding, TF lookup, message-shape conversion, and
per-second instrumentation.

History:
  - feat/30 replaced the previous fsd_path_planning-based implementation
    with a from-scratch planner that better fits this pipeline. The
    upstream library couldn't be installed cleanly from git on the
    image's setuptools/pip versions, and the algorithm we needed was
    simple enough to write directly. This is now a self-contained
    adapter: no third-party planner dep, no vendored sources.
"""

from __future__ import annotations

from typing import List

import numpy as np
import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from transforms3d.euler import quat2euler, euler2quat

from path_planning.planner import (
    Cone, ConeColor, Pose2D, plan_centerline,
)


# Canonical cone-color RGB tuples published by cone_graph_slam (see
# pipeline/cone_slam/cone_slam/cone_graph_slam_node.py:519+). The
# planner only uses BLUE and YELLOW; ORANGE and BIG_ORANGE pass through
# the classifier but are filtered out by plan_centerline.
CONE_COLOR_RGB = [
    ((1.0, 1.0, 0.0), ConeColor.YELLOW),
    ((0.0, 0.4, 1.0), ConeColor.BLUE),
    ((1.0, 0.5, 0.0), ConeColor.ORANGE),
    ((1.0, 0.3, 0.0), ConeColor.BIG_ORANGE),
]
# Per-channel L1 budget — tighter than 0.5 in any channel so a noisy
# (1.0, 0.4, 0.0) doesn't leak into BIG_ORANGE territory.
COLOR_MATCH_TOL = 0.30


def _classify_cone_color(r: float, g: float, b: float) -> ConeColor | None:
    """Return the closest canonical ConeColor, or None if unrecognized.

    `None` flags an out-of-palette marker (typically a malformed publisher
    or an UNKNOWN cone from a future SLAM revision); plan_centerline
    will skip it because only BLUE / YELLOW are used.
    """
    best = None
    best_dist = COLOR_MATCH_TOL * 3.0
    for (cr, cg, cb), color in CONE_COLOR_RGB:
        d = abs(r - cr) + abs(g - cg) + abs(b - cb)
        if d < best_dist:
            best_dist = d
            best = color
    return best


def _pose_stamped(x: float, y: float, yaw: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "odom"
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = 0.0
    qw, qx, qy, qz = euler2quat(0.0, 0.0, yaw)
    p.pose.orientation.w = float(qw)
    p.pose.orientation.x = float(qx)
    p.pose.orientation.y = float(qy)
    p.pose.orientation.z = float(qz)
    return p


class Plan_Path(Node):
    def __init__(self) -> None:
        super().__init__("Plan_Path")

        self.publisher_path = self.create_publisher(Path, "Path", 10)
        self.create_subscription(
            MarkerArray, "Conos", self._on_cones, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Per-second instrumentation. Each callback falls into exactly
        # one bucket; rates are logged every ~1 s in _maybe_log_stats.
        # In a healthy run cb≈publish; everything else is a starvation
        # signal.
        self._stats = {
            "callbacks": 0,
            "no_cones": 0,        # 0 BLUE + 0 YELLOW after color decode
            "tf_miss": 0,         # TF lookup failed
            "plan_empty": 0,      # plan_centerline returned []
            "publish": 0,
        }
        self._stats_prev = dict(self._stats)
        self._stats_last_log_ns = 0

    def _maybe_log_stats(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self._stats_last_log_ns == 0:
            self._stats_last_log_ns = now_ns
            return
        if now_ns - self._stats_last_log_ns < 1_000_000_000:
            return
        dt = (now_ns - self._stats_last_log_ns) / 1e9
        d = {k: self._stats[k] - self._stats_prev[k] for k in self._stats}
        self.get_logger().info(
            f"PATH_RATE cb={d['callbacks']/dt:4.1f}/s "
            f"pub={d['publish']/dt:4.1f}/s "
            f"no_cones={d['no_cones']} tf_miss={d['tf_miss']} "
            f"plan_empty={d['plan_empty']}"
        )
        self._stats_prev = dict(self._stats)
        self._stats_last_log_ns = now_ns

    def _on_cones(self, msg: MarkerArray) -> None:
        self._stats["callbacks"] += 1

        cones: List[Cone] = []
        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                continue
            color = _classify_cone_color(m.color.r, m.color.g, m.color.b)
            if color is None:
                continue
            cones.append(Cone(
                x=float(m.pose.position.x),
                y=float(m.pose.position.y),
                color=color,
            ))

        if not cones:
            self._stats["no_cones"] += 1
            self._maybe_log_stats()
            return

        # Pose lookup. cone_graph_slam owns odom→base_link.
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom", "base_link", rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            self._stats["tf_miss"] += 1
            self._maybe_log_stats()
            return

        yaw = quat2euler([
            tf.transform.rotation.w,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
        ])[2]
        pose = Pose2D(
            x=float(tf.transform.translation.x),
            y=float(tf.transform.translation.y),
            yaw=float(yaw),
        )

        path_points = plan_centerline(cones, pose)
        if not path_points:
            self._stats["plan_empty"] += 1
            self._maybe_log_stats()
            return

        out = Path()
        out.header.frame_id = "odom"
        for p in path_points:
            out.poses.append(_pose_stamped(p.x, p.y, p.yaw))

        self.publisher_path.publish(out)
        self._stats["publish"] += 1
        self._maybe_log_stats()


def plan_path(args=None) -> None:
    """Entry point: spin Plan_Path until SIGINT."""
    rclpy.init(args=args)
    node = Plan_Path()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# Legacy `Reset` service-client entry point. The /reset RPC was removed
# from the bridge, so this is a no-op stub kept only for setup.py
# back-compat — calling `Reset` now exits cleanly with one warning
# rather than spinning forever waiting for a non-existent service.
def reiniciar(args=None) -> None:
    rclpy.init(args=args)
    node = rclpy.create_node("path_planning_reset_stub")
    node.get_logger().warn(
        "path_planning Reset entry point is a stub — /reset is not exposed by "
        "the current bridge. No action taken.")
    node.destroy_node()
    rclpy.shutdown()
