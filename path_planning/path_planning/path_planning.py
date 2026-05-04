"""Path planning ROS 2 node — thin adapter around the FaSTTUBe planner.

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

Algorithm: delegated to `fasttube_adapter.FasttubeAdapter`, which wraps
`fsd_path_planning.PathPlanner` (FaSTTUBe / papalotis, MIT). This module
is only the ROS 2 plumbing — color decoding, TF lookup, message-shape
conversion, and per-second instrumentation.

History:
  - PR #243 replaced the in-house Delaunay+best-first walker with the
    FaSTTUBe library. The walker was poisoned by orange-classified false
    positives in the cone soup (independently tracked on fix/241) and
    struggled at one-sided observation regions (#189). FaSTTUBe sorts
    each side independently and matches across sides, eliminating both
    failure classes in one swap. The Delaunay debug overlay
    (`/path_planning/delaunay`) was dropped along with the algorithm —
    the new planner doesn't expose triangulation internals and a stale
    fake overlay would mislead more than help.
  - feat/30 had previously replaced an even older fsd_path_planning-based
    implementation with the from-scratch walker; this PR returns to the
    FaSTTUBe library now that pip install issues are resolved (pinned
    commit, --no-deps in the Dockerfile).
"""

from __future__ import annotations

import json
import os
from typing import List

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from transforms3d.euler import quat2euler, euler2quat

from path_planning.core_types import Cone, ConeColor, Pose2D
from path_planning.fasttube_adapter import FasttubeAdapter


# Canonical cone-color RGB tuples published by cone_graph_slam (see
# pipeline/cone_slam/cone_slam/cone_graph_slam_node.py:519+). The
# planner consumes BLUE, YELLOW, ORANGE, BIG_ORANGE — FaSTTUBe sorts
# yellow/blue per side and treats orange as start/finish markers.
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
    or an UNKNOWN cone from a future SLAM revision); the adapter will
    skip it because it has no FaSTTUBe ConeTypes mapping.
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

        # FaSTTUBe planner instance. Reused across cone callbacks; the
        # library is stateless for trackdrive (the only mission this
        # node currently supports — issue #243 covers skidpad/accel).
        self._adapter = FasttubeAdapter()

        # Per-second instrumentation. Each callback falls into exactly
        # one bucket; rates are logged every ~1 s in _maybe_log_stats.
        # In a healthy run cb≈publish; everything else is a starvation
        # signal.
        self._stats = {
            "callbacks": 0,
            "no_cones": 0,        # 0 cones after color decode
            "tf_miss": 0,         # TF lookup failed
            "plan_empty": 0,      # adapter returned []
            "publish": 0,
        }
        self._stats_prev = dict(self._stats)
        self._stats_last_log_ns = 0

        # Optional per-tick capture: when DV_PLANNER_CAPTURE is set to
        # a writable file path, every callback dumps (cones, pose,
        # n_path) as a JSON line. Lets us replay real failing scenes
        # through the planner offline and write regression tests with
        # actual SLAM-derived data — synthetic geometries miss the
        # frame-to-frame-instability + sparse-cone failure modes that
        # show up in PIE.
        self._capture_path = os.environ.get("DV_PLANNER_CAPTURE", "")
        self._capture_fh = None
        if self._capture_path:
            try:
                self._capture_fh = open(self._capture_path, "w")
                self.get_logger().info(
                    f"DV_PLANNER_CAPTURE → {self._capture_path}")
            except OSError as ex:
                self.get_logger().error(f"capture open failed: {ex}")
                self._capture_fh = None

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

        path_points = self._adapter.plan(cones, pose)

        # Tick capture: dump (cones, pose, n_path) for offline replay.
        if self._capture_fh is not None:
            try:
                self._capture_fh.write(json.dumps({
                    "t_ns": self.get_clock().now().nanoseconds,
                    "pose": [pose.x, pose.y, pose.yaw],
                    "cones": [[c.x, c.y, int(c.color)] for c in cones],
                    "n_path": len(path_points),
                }) + "\n")
                self._capture_fh.flush()
            except Exception:
                pass

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
