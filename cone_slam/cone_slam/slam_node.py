"""Two-phase cone SLAM lifecycle node (#496).

Replaces `cone_graph_slam_node.py`. Phase 1 builds a cone map by
accumulating observations against an external pose source (option D
of the rewrite plan); Phase 2 (next phase of the rewrite) takes the
frozen map and runs ICP/Mahalanobis localisation on it.

This file is the ROS wrapper — the per-tick logic lives in
`phase1_mapper.py` + `lap_detector.py` + `frozen_map.py`, all of
which are pure-Python and unit-testable without rclpy / GTSAM.

## Topology (Phase 1 only — Phase 2 wires in later)

Subscribes:
  /odom                     pose source (production)
  /Conos_raw                MarkerArray, cone observations (body frame)
  /Conos_Orange             MarkerArray, big-orange overlay (used for
                             lap detection — same observations as
                             /Conos_raw but tagged is_big_orange).
  /testing_only/odom        ground truth (diagnostic only; can be
                             promoted to the pose source via the
                             `pose_source` parameter for offline tests
                             that don't trust /odom).

Publishes:
  /slam/pose                nav_msgs/Odometry in map frame.
  /Conos                    MarkerArray of the live landmark map.
  /cone_slam/gt_aligned     GT pose re-anchored into SLAM's frame
                             (the same diagnostic the old
                             cone_graph_slam_node emitted).
  /cone_slam/gt_error_m     scalar SLAM-vs-GT residual.
  /slam/finished            Bool (latched), true on lap completion
                             for single-lap modes (autocross etc.).

Broadcasts:
  map → odom                 Identity for Phase 1 (no drift correction
                             needed — pose source IS odom, so the two
                             frames are coincident). Phase 2 will
                             populate this with a real
                             `slam_pose ⊖ /odom` transform.

## ~/setup behavior dispatch

The `behavior` field on /setup picks Phase 1's mode-specific
parameters (lap_count_target, lap-distance gate, etc.). See
`_select_dispatcher`.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.lifecycle import State, TransitionCallbackReturn
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, ColorRGBA, Float32
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from node_base.base_lifecycle_node import BaseLifecycleNode

from cone_slam.frozen_map import FrozenMap
from cone_slam.landmark_db import LandmarkDb
from cone_slam.lap_detector import LapDetector
from cone_slam.phase1_mapper import Observation, Phase1Mapper, Pose2D
from cone_slam.phase2_localiser import Phase2Localiser


# Per-behavior lap-completion parameters. Phase 1's mapper is the
# same across modes; only the lap detector + finish-trigger shape
# differs.
_BEHAVIOR_PARAMS: dict[str, dict] = {
    # trackdrive: multi-lap. Phase 2 takes over after the FIRST
    # crossing; lap_count_target gates the /slam/finished publish.
    "trackdrive": {
        "lap_count_target": 10,
        "uses_lap_gate": True,
        "min_lap_distance_m": 30.0,
        "single_lap_finish": False,
    },
    # autocross: single lap. /slam/finished fires on first crossing.
    "autocross": {
        "lap_count_target": 1,
        "uses_lap_gate": True,
        "min_lap_distance_m": 30.0,
        "single_lap_finish": True,
    },
    # acceleration: 75 m straight line. No big-orange gate — finish
    # is purely distance-based.
    "accel": {
        "lap_count_target": 1,
        "uses_lap_gate": False,
        "min_lap_distance_m": 75.0,
        "single_lap_finish": True,
    },
    # skidpad: 4-lap figure-eight. Big-orange gate exists at the
    # entry/exit. Treat as "trackdrive with lap_count_target=4" for
    # Phase 1; Phase 2 may add geometry-specific logic later.
    "skidpad": {
        "lap_count_target": 4,
        "uses_lap_gate": True,
        "min_lap_distance_m": 15.0,
        "single_lap_finish": False,
    },
    # Default fallback (scruti, anything else): Phase 1 only, never
    # auto-finish — the operator stops the session manually.
    "default": {
        "lap_count_target": None,
        "uses_lap_gate": False,
        "min_lap_distance_m": 30.0,
        "single_lap_finish": False,
    },
}


class SlamNode(BaseLifecycleNode):
    """Phase 1 cone-mapping lifecycle node.

    Phase 2 localisation will subclass / extend this once
    `phase2_localiser.py` lands; for now the node terminates at
    Phase 1 and stays in mapping mode for the whole mission.
    """

    NODE_NAME = "slam_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        # Parameters declared up-front so they're visible via
        # `ros2 param list` even before on_configure runs.
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # Pose-source selector. "odom" (production) or "gt" (replay
        # / regression tests that don't trust /odom drift). The
        # `gt` mode subscribes to /testing_only/odom AS the pose
        # source rather than as a diagnostic feed.
        self.declare_parameter("pose_source", "odom")
        # Phase 1 algorithm knobs.
        self.declare_parameter("da_gate_m", 1.0)
        self.declare_parameter("obs_sigma_m", 0.20)
        # Phase 2 algorithm knobs. Take effect at the MAPPING →
        # LOCALISING transition (first lap completion in multi-lap
        # modes, or `phase2_force_freeze_after_n_scans` for testing).
        self.declare_parameter("phase2_obs_sigma_m", 0.20)
        self.declare_parameter("phase2_max_match_radius_m", 3.0)
        # Process-noise σ applied per scan-tick in Phase 2's predict
        # step. Defaults reflect the per-tick (not per-second) drift
        # an /odom-fed body delta accumulates on FS-DV trackdrives.
        self.declare_parameter("phase2_proc_sigma_xy_m", 0.05)
        self.declare_parameter("phase2_proc_sigma_yaw_deg", 0.5)
        # Test/diagnostic knob: force the freeze + Phase 2 switchover
        # after N cone scans, ignoring the lap detector. 0 disables
        # the override and the node uses the natural lap-gated path.
        self.declare_parameter("phase2_force_freeze_after_n_scans", 0)

        # Runtime state — set up in on_configure_impl.
        self._db: Optional[LandmarkDb] = None
        self._mapper: Optional[Phase1Mapper] = None
        self._lap_detector: Optional[LapDetector] = None
        self._dispatcher: Optional[dict] = None
        self._lap_count: int = 0

        self._latest_pose: Optional[Pose2D] = None
        self._latest_odom_msg: Optional[Odometry] = None
        self._latest_gt_msg: Optional[Odometry] = None
        self._gt_init_pose: Optional[Pose2D] = None

        # Phase 2 state. `_localiser` is None while we're mapping;
        # populated by `_transition_to_phase2` at the freeze point.
        # `_last_pose_for_delta` caches the previous /odom-frame
        # pose so Phase 2's predict step can use a body-frame delta.
        self._localiser: Optional[Phase2Localiser] = None
        self._last_pose_for_delta: Optional[Pose2D] = None
        # Total cone scans processed, regardless of phase. Mapper's
        # internal step counter stops advancing after the freeze, so
        # we keep this separate for diagnostics + the replay snapshot.
        self._total_scans: int = 0

        # Subscriptions / publishers — populated in on_configure /
        # on_activate.
        self._sub_odom = None
        self._sub_cones = None
        self._sub_orange = None
        self._sub_gt = None
        self._pose_pub = None
        self._cones_pub = None
        self._gt_aligned_pub = None
        self._gt_error_pub = None
        self._finished_pub = None
        self._tf_broadcaster: Optional[TransformBroadcaster] = None
        self._finished_emitted = False
        # Cache the "is this observation big-orange" set so the
        # mapper can tag landmarks correctly. Big-orange comes from
        # cone_detection on a separate topic /Conos_Orange, indexed
        # by marker.id; we map id → True at marker arrival.
        self._big_orange_ids: set[int] = set()

    # ----- lifecycle -----

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        ret = super().on_configure(state)
        if ret != TransitionCallbackReturn.SUCCESS:
            return ret

        self._map_frame = self.get_parameter("map_frame").value
        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._pose_source = self.get_parameter("pose_source").value
        da_gate_m = float(self.get_parameter("da_gate_m").value)
        obs_sigma_m = float(self.get_parameter("obs_sigma_m").value)

        # Phase 2 knobs cached for `_transition_to_phase2`.
        self._phase2_obs_sigma_m = float(
            self.get_parameter("phase2_obs_sigma_m").value)
        self._phase2_max_match_radius_m = float(
            self.get_parameter("phase2_max_match_radius_m").value)
        self._phase2_proc_sigma_xy_m = float(
            self.get_parameter("phase2_proc_sigma_xy_m").value)
        self._phase2_proc_sigma_yaw_rad = math.radians(float(
            self.get_parameter("phase2_proc_sigma_yaw_deg").value))
        self._phase2_force_freeze_after_n_scans = int(
            self.get_parameter("phase2_force_freeze_after_n_scans").value)

        # Initialise Phase 1 components.
        self._db = LandmarkDb()
        self._mapper = Phase1Mapper(
            self._db,
            da_gate_m=da_gate_m,
            obs_sigma_m=obs_sigma_m,
        )
        self._dispatcher = _BEHAVIOR_PARAMS.get(
            self.behavior or "default", _BEHAVIOR_PARAMS["default"],
        )
        if self._dispatcher["uses_lap_gate"]:
            self._lap_detector = LapDetector(
                min_lap_distance_m=self._dispatcher["min_lap_distance_m"],
            )
        else:
            self._lap_detector = None

        # Publishers (lifecycle — silent until activate).
        self._pose_pub = self.create_lifecycle_publisher(
            Odometry, "/slam/pose", 10,
        )
        self._cones_pub = self.create_lifecycle_publisher(
            MarkerArray, "/Conos", 10,
        )
        self._gt_aligned_pub = self.create_lifecycle_publisher(
            Odometry, "/cone_slam/gt_aligned", 10,
        )
        self._gt_error_pub = self.create_lifecycle_publisher(
            Float32, "/cone_slam/gt_error_m", 10,
        )
        # /slam/finished is latched + transient_local so a
        # late-joining mission_control sees the most recent value.
        finished_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._finished_pub = self.create_lifecycle_publisher(
            Bool, "/slam/finished", finished_qos,
        )
        self._tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"on_configure: behavior={self.behavior!r} "
            f"pose_source={self._pose_source!r} "
            f"da_gate_m={da_gate_m} obs_sigma_m={obs_sigma_m} "
            f"lap_count_target={self._dispatcher['lap_count_target']}"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        # Reset per-mission state.
        self._lap_count = 0
        self._finished_emitted = False
        self._latest_pose = None
        self._latest_odom_msg = None
        self._latest_gt_msg = None
        self._gt_init_pose = None
        self._big_orange_ids = set()
        self._localiser = None
        self._last_pose_for_delta = None
        self._total_scans = 0
        if hasattr(self, "_slam_init_pose"):
            self._slam_init_pose = None

        # Subscriptions. QoS rationale:
        #   /odom              — odometry_filter publishes RELIABLE
        #                        (post-#526), 100 Hz; match it.
        #   /testing_only/odom — the bridge publishes BEST_EFFORT
        #                        (see ifssim_ros_wrapper.cpp:sensor_qos)
        #                        and rosbag2 replays preserve the
        #                        recorded QoS. Subscribing RELIABLE
        #                        triggers "incompatible QoS" warnings
        #                        and no messages arrive. Match the
        #                        publisher with BEST_EFFORT.
        odom_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        gt_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        # Pose source selection. In production we read /odom; in
        # replay / regression mode we promote /testing_only/odom to
        # the same role so the mapper is tested against GT-quality
        # pose. /odom keeps being subscribed in BOTH modes so the
        # map → odom TF can be published with the right anchor.
        if self._pose_source == "gt":
            self._sub_gt = self.create_subscription(
                Odometry, "/testing_only/odom",
                self._on_gt_as_pose_source, gt_qos,
            )
            self.get_logger().info(
                "pose_source=gt — promoting /testing_only/odom to the "
                "Phase 1 pose feed (testing mode)."
            )
        else:
            self._sub_odom = self.create_subscription(
                Odometry, "/odom", self._on_odom, odom_qos,
            )
            self._sub_gt = self.create_subscription(
                Odometry, "/testing_only/odom",
                self._on_gt_odom, gt_qos,
            )

        cones_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._sub_cones = self.create_subscription(
            MarkerArray, "/Conos_raw", self._on_cones, cones_qos,
        )
        self._sub_orange = self.create_subscription(
            MarkerArray, "/Conos_Orange", self._on_orange, cones_qos,
        )

        # Latched-false initial state for /slam/finished — readers
        # need a defined starting value.
        self._publish_finished(False)

        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        for sub in (self._sub_odom, self._sub_cones, self._sub_orange,
                    self._sub_gt):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_odom = self._sub_cones = self._sub_orange = self._sub_gt = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        for pub in (self._pose_pub, self._cones_pub, self._gt_aligned_pub,
                    self._gt_error_pub, self._finished_pub):
            if pub is not None:
                self.destroy_publisher(pub)
        self._pose_pub = self._cones_pub = None
        self._gt_aligned_pub = self._gt_error_pub = self._finished_pub = None
        self._tf_broadcaster = None
        self._db = None
        self._mapper = None
        self._lap_detector = None
        return super().on_cleanup(state)

    # ----- callbacks -----

    def _on_odom(self, msg: Odometry) -> None:
        """Cache the latest production /odom pose."""
        self._latest_odom_msg = msg
        self._latest_pose = Pose2D.from_ros_pose(msg.pose.pose)

    def _on_gt_as_pose_source(self, msg: Odometry) -> None:
        """Replay/test mode: /testing_only/odom IS the pose feed."""
        self._latest_odom_msg = msg
        self._latest_pose = Pose2D.from_ros_pose(msg.pose.pose)
        # Also fill GT cache so the GT-aligned diagnostic still
        # emits — it'll just compare against itself (residual ≈ 0).
        self._latest_gt_msg = msg

    def _on_gt_odom(self, msg: Odometry) -> None:
        """Production: /testing_only/odom is diagnostic only."""
        self._latest_gt_msg = msg

    def _on_orange(self, msg: MarkerArray) -> None:
        """Build the set of marker IDs that are big-orange in this
        scan. The `/Conos_Orange` topic publishes the same scan as
        /Conos_raw but containing only the big-orange-tagged subset;
        we use it as a side channel for the is_big_orange flag.
        Indexed by marker.id, which cone_detection holds stable
        across topics within a single scan."""
        ids = set()
        for m in msg.markers:
            ids.add(int(m.id))
        self._big_orange_ids = ids

    def _on_cones(self, msg: MarkerArray) -> None:
        """Per-scan tick. Branches on whether Phase 1 (mapping) or
        Phase 2 (localising) owns the current state."""
        if self._mapper is None or self._latest_pose is None:
            return

        observations: list[Observation] = []
        for m in msg.markers:
            observations.append(Observation(
                body_x=float(m.pose.position.x),
                body_y=float(m.pose.position.y),
                is_big_orange=int(m.id) in self._big_orange_ids,
            ))

        # Count the scan regardless of whether it carried any
        # observations — empty scans still represent time / motion
        # the filter should account for.
        self._total_scans += 1

        if self._localiser is None:
            # Phase 1: nothing to do on an empty scan (the mapper
            # has no internal state to advance).
            if observations:
                self._tick_phase1(msg, observations)
        else:
            # Phase 2: always tick. predict() must run every scan
            # to keep the pose advancing; update() is skipped if
            # the scan was empty.
            self._tick_phase2(msg, observations)

    def _tick_phase1(self, msg: MarkerArray,
                     observations: list[Observation]) -> None:
        """Phase 1 (mapping) per-scan body."""
        summary = self._mapper.observe_scan(self._latest_pose, observations)

        self._publish_slam_pose(msg)
        self._publish_landmarks(msg)
        self._publish_map_to_odom_tf(msg)
        self._publish_gt_aligned(msg)

        self._update_lap_state()

        if summary["step"] % 50 == 0:
            self.get_logger().info(
                f"SLAM_OBS step={summary['step']} "
                f"obs={summary['n_obs']} assoc={summary['n_assoc']} "
                f"new={summary['n_new']} map={summary['n_map']} "
                f"big_orange={summary['n_big_orange']} "
                f"laps={self._lap_count}"
            )

        # Possible Phase 1 → Phase 2 freeze. Order matters: the
        # decision uses the lap state we just updated.
        if self._should_freeze_now(summary["step"]):
            self._transition_to_phase2()

    def _tick_phase2(self, msg: MarkerArray,
                     observations: list[Observation]) -> None:
        """Phase 2 (localising) per-scan body. Predict from a
        body-frame delta of the raw pose feed, update against the
        frozen map (when there are observations), publish the
        corrected pose."""
        assert self._localiser is not None
        # Body-frame delta from previous /odom-frame pose. Runs
        # every scan whether or not observations are present —
        # otherwise the pose freezes during cone-less windows
        # (turn-aways, lidar occlusion, etc.).
        if self._last_pose_for_delta is not None:
            prev = self._last_pose_for_delta
            curr = self._latest_pose
            dxw = curr.x - prev.x
            dyw = curr.y - prev.y
            dtheta = _wrap_pi(curr.yaw - prev.yaw)
            c, s = math.cos(-prev.yaw), math.sin(-prev.yaw)
            dx_body = c * dxw - s * dyw
            dy_body = s * dxw + c * dyw
            self._localiser.predict(
                dx_body=dx_body, dy_body=dy_body, dtheta=dtheta,
                sigma_xy=self._phase2_proc_sigma_xy_m,
                sigma_yaw=self._phase2_proc_sigma_yaw_rad,
            )
        self._last_pose_for_delta = self._latest_pose

        if observations:
            summary = self._localiser.update(observations)
        else:
            summary = None

        # Publish corrected pose + frozen map (unchanged) + TF.
        self._publish_phase2_pose(msg)
        self._publish_landmarks(msg)
        self._publish_phase2_tf(msg)
        self._publish_gt_aligned(msg)

        # Lap state — keep counting so multi-lap modes can hit
        # their target. The detector still reads landmarks from
        # `_db`; the frozen map is the same memory.
        self._update_lap_state()

        if summary is not None and summary.n_obs \
                and self._total_scans % 50 == 0:
            self.get_logger().info(
                f"PHASE2 scan={self._total_scans} "
                f"obs={summary.n_obs} matched={summary.n_matched} "
                f"gated={summary.n_gated_out} "
                f"unmatched={summary.n_unmatched} "
                f"mean_innov={summary.mean_innovation_m:.2f}m "
                f"laps={self._lap_count}"
            )

    # ----- Phase 1 → Phase 2 transition -----

    def _should_freeze_now(self, step: int) -> bool:
        """Decide whether to take the snapshot and switch to Phase 2.

        Two triggers, in priority order:
          1. `phase2_force_freeze_after_n_scans` > 0 — testing
             override; switch deterministically at scan N regardless
             of lap state. Lets the regression suite exercise Phase
             2 on bags that don't contain a real lap-completion event.
          2. Lap detector fired (and we're in a multi-lap mode).
             Single-lap modes (autocross, accel) have nothing left
             to do after the first crossing, so Phase 2 isn't worth
             paying the predict/update cost for.
        """
        force_n = self._phase2_force_freeze_after_n_scans
        if force_n > 0 and step >= force_n:
            return True
        if self._dispatcher is None:
            return False
        if self._dispatcher.get("single_lap_finish"):
            return False
        return self._lap_count >= 1

    def _transition_to_phase2(self) -> None:
        """Snapshot the live map, hand it to a fresh Phase2Localiser,
        seed it with the current pose. The mapper stops being driven
        from this point — the LandmarkDb is kept around only so
        downstream consumers (lap detector, /Conos publishing) keep
        seeing the same cones."""
        if self._mapper is None or self._latest_pose is None:
            return
        frozen = FrozenMap.from_landmarks(self._mapper.snapshot_for_freeze())
        self._localiser = Phase2Localiser(
            frozen,
            self._latest_pose,
            obs_sigma_m=self._phase2_obs_sigma_m,
            max_match_radius_m=self._phase2_max_match_radius_m,
        )
        self._last_pose_for_delta = self._latest_pose
        self.get_logger().info(
            f"PHASE2_ACTIVE frozen_cones={len(frozen)} "
            f"init_pose=({self._latest_pose.x:.2f}, "
            f"{self._latest_pose.y:.2f}, "
            f"{math.degrees(self._latest_pose.yaw):.1f}°)"
        )

    # ----- publishing -----

    def _publish_slam_pose(self, scan_msg: MarkerArray) -> None:
        if self._latest_odom_msg is None:
            return
        out = Odometry()
        out.header.stamp = scan_msg.markers[0].header.stamp if scan_msg.markers \
            else self.get_clock().now().to_msg()
        out.header.frame_id = self._map_frame
        out.child_frame_id = self._base_frame
        out.pose = self._latest_odom_msg.pose
        out.twist = self._latest_odom_msg.twist
        self._pose_pub.publish(out)

    def _publish_landmarks(self, scan_msg: MarkerArray) -> None:
        ma = MarkerArray()
        stamp = scan_msg.markers[0].header.stamp if scan_msg.markers \
            else self.get_clock().now().to_msg()
        for lm in self._db:
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp = stamp
            m.ns = "cone_slam"
            m.id = lm.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(lm.position[0])
            m.pose.position.y = float(lm.position[1])
            m.pose.position.z = float(lm.position[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.30
            m.scale.y = 0.30
            m.scale.z = 0.32
            # Big-orange cones rendered orange, others yellow-ish
            # to match the convention. Phase 1 doesn't infer L/R
            # colour from observations (that's a perception
            # problem, not a SLAM problem).
            if lm.is_big_orange:
                m.color = ColorRGBA(r=1.0, g=0.45, b=0.0, a=0.9)
            else:
                m.color = ColorRGBA(r=0.95, g=0.95, b=0.1, a=0.7)
            ma.markers.append(m)
        self._cones_pub.publish(ma)

    def _publish_map_to_odom_tf(self, scan_msg: MarkerArray) -> None:
        """Identity transform during Phase 1.

        Phase 1's pose source is `/odom` (or GT, in replay), so
        SLAM's `map` frame is just `odom` with no drift correction.
        Publish identity so the TF tree resolves end-to-end for
        downstream consumers (path_planning, control). Phase 2
        overrides this via `_publish_phase2_tf`.
        """
        if self._tf_broadcaster is None:
            return
        t = TransformStamped()
        t.header.stamp = scan_msg.markers[0].header.stamp if scan_msg.markers \
            else self.get_clock().now().to_msg()
        t.header.frame_id = self._map_frame
        t.child_frame_id = self._odom_frame
        t.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(t)

    def _publish_phase2_pose(self, scan_msg: MarkerArray) -> None:
        """Publish the EKF-corrected pose as /slam/pose. Twist is
        copied from /odom — Phase 2 doesn't estimate velocity, only
        pose; downstream consumers that need vx/vy still trust the
        odometry filter for those."""
        if self._localiser is None:
            return
        pose = self._localiser.pose
        out = Odometry()
        out.header.stamp = scan_msg.markers[0].header.stamp if scan_msg.markers \
            else self.get_clock().now().to_msg()
        out.header.frame_id = self._map_frame
        out.child_frame_id = self._base_frame
        out.pose.pose.position.x = pose.x
        out.pose.pose.position.y = pose.y
        half = 0.5 * pose.yaw
        out.pose.pose.orientation.w = math.cos(half)
        out.pose.pose.orientation.z = math.sin(half)
        if self._latest_odom_msg is not None:
            out.twist = self._latest_odom_msg.twist
        self._pose_pub.publish(out)

    def _publish_phase2_tf(self, scan_msg: MarkerArray) -> None:
        """Compute and broadcast the real `map → odom` transform.

        With `T_map_base = SLAM pose` and `T_odom_base = /odom pose`,
        the SE(2) algebra gives:
            T_map_odom = T_map_base ⊖ T_odom_base
        which lets downstream nodes that still publish in `odom`
        (path planning, control) project into `map` consistently.
        """
        if self._tf_broadcaster is None or self._localiser is None \
                or self._latest_pose is None:
            return
        slam = self._localiser.pose
        odom = self._latest_pose
        dtheta = _wrap_pi(slam.yaw - odom.yaw)
        c, s = math.cos(slam.yaw), math.sin(slam.yaw)
        co, so = math.cos(odom.yaw), math.sin(odom.yaw)
        # base in odom: (odom.x, odom.y); we want
        # T_map_odom.position = slam.position - R(dtheta) @ odom.position
        dx = slam.x - (math.cos(dtheta) * odom.x - math.sin(dtheta) * odom.y)
        dy = slam.y - (math.sin(dtheta) * odom.x + math.cos(dtheta) * odom.y)
        t = TransformStamped()
        t.header.stamp = scan_msg.markers[0].header.stamp if scan_msg.markers \
            else self.get_clock().now().to_msg()
        t.header.frame_id = self._map_frame
        t.child_frame_id = self._odom_frame
        t.transform.translation.x = dx
        t.transform.translation.y = dy
        half = 0.5 * dtheta
        t.transform.rotation.w = math.cos(half)
        t.transform.rotation.z = math.sin(half)
        self._tf_broadcaster.sendTransform(t)

    def _effective_pose(self) -> Optional[Pose2D]:
        """SLAM's externally-published pose: localiser output when
        Phase 2 is active, raw pose feed when Phase 1 is."""
        if self._localiser is not None:
            return self._localiser.pose
        return self._latest_pose

    def _publish_gt_aligned(self, scan_msg: MarkerArray) -> None:
        """GT pose re-anchored to SLAM's calibration-end frame —
        ports `cone_graph_slam_node._publish_gt_aligned`."""
        slam_pose = self._effective_pose()
        if self._latest_gt_msg is None or slam_pose is None:
            return
        gt_now = Pose2D.from_ros_pose(self._latest_gt_msg.pose.pose)

        # Anchor: take the first sample where BOTH gt and slam are
        # available, so the two re-anchored trajectories start from
        # the same instant (avoids the constant ~1 m bias from
        # snapshotting them at different scans — see SLAM rewrite
        # caveat #2).
        if self._gt_init_pose is None:
            self._gt_init_pose = gt_now
            self._slam_init_pose = slam_pose
            return

        # Express gt_now in the anchor frame.
        dx = gt_now.x - self._gt_init_pose.x
        dy = gt_now.y - self._gt_init_pose.y
        c = math.cos(-self._gt_init_pose.yaw)
        s = math.sin(-self._gt_init_pose.yaw)
        gt_aligned_x = c * dx - s * dy
        gt_aligned_y = s * dx + c * dy
        gt_aligned_yaw = _wrap_pi(gt_now.yaw - self._gt_init_pose.yaw)

        # Same anchor math applied to SLAM's pose.
        slam_init = self._slam_init_pose
        sx = slam_pose.x - slam_init.x
        sy = slam_pose.y - slam_init.y
        c = math.cos(-slam_init.yaw)
        s = math.sin(-slam_init.yaw)
        slam_aligned_x = c * sx - s * sy
        slam_aligned_y = s * sx + c * sy

        out = Odometry()
        out.header.stamp = scan_msg.markers[0].header.stamp if scan_msg.markers \
            else self.get_clock().now().to_msg()
        out.header.frame_id = self._map_frame
        out.pose.pose.position.x = gt_aligned_x
        out.pose.pose.position.y = gt_aligned_y
        half = 0.5 * gt_aligned_yaw
        out.pose.pose.orientation.w = math.cos(half)
        out.pose.pose.orientation.z = math.sin(half)
        self._gt_aligned_pub.publish(out)

        err_m = math.hypot(slam_aligned_x - gt_aligned_x,
                            slam_aligned_y - gt_aligned_y)
        self._gt_error_pub.publish(Float32(data=float(err_m)))

    def _publish_finished(self, value: bool) -> None:
        if self._finished_pub is None:
            return
        self._finished_pub.publish(Bool(data=bool(value)))

    # ----- lap accounting -----

    def _update_lap_state(self) -> None:
        pose = self._effective_pose()
        if self._lap_detector is None or pose is None:
            return
        landmarks = list(self._db) if self._db is not None else []
        crossed = self._lap_detector.observe(pose, landmarks)
        if crossed:
            self._lap_count += 1
            self.get_logger().info(
                f"LAP_COMPLETED count={self._lap_count} "
                f"distance={self._lap_detector.cumulative_distance_m:.1f}m"
            )
            target = self._dispatcher.get("lap_count_target")
            if (self._dispatcher.get("single_lap_finish")
                    or (target is not None and self._lap_count >= target)):
                if not self._finished_emitted:
                    self._publish_finished(True)
                    self._finished_emitted = True
                    self.get_logger().info("/slam/finished -> true")
            else:
                # Multi-lap mode (trackdrive): re-arm for next lap.
                self._lap_detector.reset_for_next_lap()

    # ----- Phase 2 handoff (stub for the next phase) -----

    def snapshot_for_phase2(self) -> Optional[FrozenMap]:
        """Build a FrozenMap from the live LandmarkDb. Phase 2 will
        call this on lap completion to take over localisation. Stub
        for now — Phase 1 alone doesn't fire it."""
        if self._mapper is None:
            return None
        return FrozenMap.from_landmarks(self._mapper.snapshot_for_freeze())

    # ----- replay / regression interface -----
    #
    # The offline replay harness (`scripts/replay_slam.py`) drives this
    # node deterministically over a recorded bag. These helpers let it
    # do that without touching private attributes:
    #
    #   REPLAY_TOPICS         — which bag topics are required.
    #   replay_setup(beh)     — bypass the ROS `~/setup` service.
    #   replay_dispatch(t, m) — route a deserialized msg to the right
    #                           callback; returns True iff the message
    #                           triggers a scan-tick (the harness uses
    #                           that to record a residual sample).
    #   replay_snapshot()     — uniform SLAM-vs-GT residual dict for
    #                           CSV / threshold checks. None when not
    #                           enough state has been seen yet.

    REPLAY_TOPICS: frozenset = frozenset({
        # Phase 1 only needs cones (raw + orange tag) and a pose feed.
        # /odom is the production source; /testing_only/odom is both
        # the gt-mode pose source AND the regression-test GT anchor.
        "/Conos_raw",
        "/Conos_Orange",
        "/odom",
        "/testing_only/odom",
    })

    def replay_setup(self, behavior: str, mode_name: str = "") -> None:
        """Inject mode_name/behavior bypassing the ROS Setup service.
        Must be called before on_configure() in replay mode."""
        self._mode_name = mode_name or behavior
        self._behavior = behavior

    def replay_dispatch(self, topic: str, msg) -> bool:
        """Route a bag message to the right callback.

        Returns True when the callback corresponds to a scan tick
        (i.e. the moment a residual should be recorded). For Phase 1
        that's exactly `/Conos_raw`."""
        if topic == "/Conos_raw":
            self._on_cones(msg)
            return True
        if topic == "/Conos_Orange":
            self._on_orange(msg)
            return False
        if topic == "/odom":
            if self._pose_source == "odom":
                self._on_odom(msg)
            return False
        if topic == "/testing_only/odom":
            if self._pose_source == "gt":
                self._on_gt_as_pose_source(msg)
            else:
                self._on_gt_odom(msg)
            return False
        return False

    def replay_snapshot(self) -> Optional[dict]:
        """SLAM-vs-GT residual snapshot in the same anchor frame as
        `_publish_gt_aligned`. Returns None until both SLAM and GT
        anchors are populated."""
        slam_pose = self._effective_pose()
        if (slam_pose is None
                or self._latest_gt_msg is None
                or self._gt_init_pose is None):
            return None
        slam_init = getattr(self, "_slam_init_pose", None)
        if slam_init is None:
            return None

        gt_now = Pose2D.from_ros_pose(self._latest_gt_msg.pose.pose)
        dx = gt_now.x - self._gt_init_pose.x
        dy = gt_now.y - self._gt_init_pose.y
        c = math.cos(-self._gt_init_pose.yaw)
        s = math.sin(-self._gt_init_pose.yaw)
        gt_aligned_x = c * dx - s * dy
        gt_aligned_y = s * dx + c * dy
        gt_aligned_yaw = _wrap_pi(gt_now.yaw - self._gt_init_pose.yaw)

        sx = slam_pose.x - slam_init.x
        sy = slam_pose.y - slam_init.y
        c = math.cos(-slam_init.yaw)
        s = math.sin(-slam_init.yaw)
        slam_aligned_x = c * sx - s * sy
        slam_aligned_y = s * sx + c * sy
        slam_aligned_yaw = _wrap_pi(slam_pose.yaw - slam_init.yaw)

        return {
            "step": self._total_scans,
            "slam_x": slam_aligned_x,
            "slam_y": slam_aligned_y,
            "slam_yaw": slam_aligned_yaw,
            "gt_x": gt_aligned_x,
            "gt_y": gt_aligned_y,
            "gt_yaw": gt_aligned_yaw,
            "n_map": len(self._db) if self._db is not None else 0,
            "phase2": self._localiser is not None,
        }


def _wrap_pi(angle: float) -> float:
    """Wrap to (-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main() -> None:
    rclpy.init()
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
