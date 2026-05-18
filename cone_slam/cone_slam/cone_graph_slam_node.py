"""Cone-graph SLAM node — PR B (cone observation factors + DA).

Subscribes:
    /imu               (sensor_msgs/Imu, BEST_EFFORT, ~400 Hz)
    /Conos_raw         (visualization_msgs/MarkerArray, base_link
                       frame, 10 Hz from Cone_Detection)
    /odom              (nav_msgs/Odometry from sim_supervisor, 100 Hz)
                       Phase 2 (#382): used to compute map→odom drift
                       correction. slam_node's pose is absolute (map
                       frame); supervisor's /odom is dead-reckoning
                       (odom frame); the difference is the drift to
                       absorb into the map→odom transform.

Publishes:
    /tf                (map -> odom, dynamic drift correction
                       computed as slam_pose ⊖ latest /odom; replaces
                       the pre-#382 odom→base_link broadcast which
                       supervisor now owns)
    /slam/pose         (nav_msgs/Odometry — absolute pose+velocity in
                       map frame; renamed from /cone_slam/state in
                       #382 so the topic name doesn't lock us into
                       "cone_slam" when we eventually swap iSAM2
                       backends)
    /Conos             (visualization_msgs/MarkerArray, MAP-frame
                       cone map with persistent landmark IDs encoded
                       as marker.id; was odom-frame pre-#382)

State machine:
    INIT_WAITING_IMU  → INIT_CALIBRATING (3 s) → SLAM_RUNNING

Lifecycle:
    INIT_CALIBRATING expects the car to be stationary. We accumulate
    IMU samples, then estimate accel/gyro biases as the mean (with
    gravity assumption (0,0,-9.81) for accel). Anchor x_0 at world
    origin and start the factor graph.

    SLAM_RUNNING: every LiDAR header.stamp triggers an IMU
    preintegration window since the last trigger, an ImuFactor between
    the prev pose and a new pose, and an iSAM2 update. PR B adds cone
    factors. PR C adds GPS.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

import rclpy
from node_base.base_lifecycle_node import BaseLifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn, State as LifecycleState
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

import gtsam
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

# color_classifier deleted: SLAM is position-only, no per-cone colour
# anywhere. Visualization renders all landmarks with the same colour.
from cone_slam.data_association import DISTANCE_GATE_M, Observation, associate
from cone_slam.factor_graph import FactorGraph, ScanResult
from cone_slam.imu_preintegrator import ImuPreintegrator, ImuSample
from cone_slam.landmark_db import LandmarkDb


def _odom_to_pose3(msg: Odometry) -> "gtsam.Pose3":
    """Convert nav_msgs/Odometry pose into gtsam.Pose3."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z),
        np.array([p.x, p.y, p.z]),
    )


# compute_map_to_odom lives in cone_slam.tf_math (pure-Python helper,
# no rclpy import, unit-testable). Re-imported here for the call site.
from cone_slam.tf_math import compute_map_to_odom  # noqa: E402, F401


CALIBRATION_SECONDS = 3.0

# Observations beyond this body-frame range are dropped before reaching
# data association. Universal practice across FSD teams (EUFS 20 m,
# QUTMS 25 m, MUR ~25 m, KIT19d 42 m): far cones have low point counts
# (Hesai ATX gives 3-5 rays at 25 m, ~30+ at 5 m), so centroid noise
# dominates and the bearing-range factor stops being informative. With
# the constant 0.20 m sigma we previously reported, a noisy 30 m cone
# was actively dragging the optimizer toward a wrong rotation — that's
# the late-drive yaw snap we kept hitting on 2026-04-27.
MAX_OBSERVATION_RANGE_M = 25.0


def cascade_spike_triggered(
    n_new: int,
    total: int,
    step: int,
    *,
    pct_min_obs: int = 5,
    pct_threshold: float = 0.60,
    count_threshold: int = 3,
    discovery_step_floor: int = 30,
) -> tuple[bool, list[str]]:
    """Pure-function expression of the two DA-failure spike gates.

    Returns (triggered, reasons). Both gates require the graph to be
    past the early-discovery phase (``step > discovery_step_floor``)
    so the local cone map is mature and almost every observation
    *should* be a re-association of a known landmark.

      * **Percentage gate**: ``n_new / total > pct_threshold`` once
        ``total >= pct_min_obs``. Catches "everything looks new" bursts
        — the historical cascade signature, see #441.
      * **Count gate**: ``n_new >= count_threshold`` AND
        ``n_associated == 0`` (i.e. ``total == n_new``). The zero-
        association requirement is what discriminates a true cascade
        (pose has drifted off the map → nothing matches) from
        legitimate cornering discovery (new cones swing into the LiDAR
        FoV but several previously-mapped ones still associate). The
        2026-05-12 live test of the initial assoc-agnostic count gate
        showed it firing on `obs=12 new=7 assoc=5` during the first
        turn and starving SLAM into IMU-only drift → off-track. Set
        ``count_threshold = 0`` to disable.

    Both gates are AND'd with the discovery-step floor.
    Extracted from cone_graph_slam_node._on_cones for unit-testability
    (test_cascade_spike_detector.py).
    """
    if step <= discovery_step_floor:
        return False, []
    reasons: list[str] = []
    if total >= pct_min_obs and n_new > int(pct_threshold * total):
        reasons.append(f"pct>{int(pct_threshold * 100)}%")
    if count_threshold > 0 and n_new >= count_threshold and n_new == total:
        reasons.append(f"n_new≥{count_threshold}&assoc=0")
    return bool(reasons), reasons


# Motor-RPM → body-frame longitudinal velocity.
#
# 2026-05-10 re-derivation (issue #380, supersedes the 2026-04-28
# value of 0.00898): bagged /odom (sim_supervisor's IMU+RPM filter
# output) and /testing_only/odom over a 41 s motion window on
# test_submodule.csv, paired samples within ±50 ms, observed
# mean(|GT.vx|) / mean(|filter.vx|) = 0.9140 — implying the previous
# 0.00898 produces a +9.4 % vx overestimate. The previous April
# diagnostic compared raw RPM×const directly to GT; today's compares
# filter-output to GT, which is the closed-loop quantity consumers
# (including this node, via the velocity prior) actually see.
#
# The new value 0.00898 × 0.9140 = 0.00821 recovers exactly the
# pure-geometry doc-derived form from docs/dv_pipeline_rebuild.md
# §3.5:
#     RPM_TO_MS = (2π × WheelRadius / GearRatio) / 60
#               = (2π × 0.228 / 2.909) / 60 = 0.00821
#
# Keeping the constant in sync with sim_supervisor's OdometryFilter
# (single source of truth — both consumers depend on the same
# value).
RPM_TO_MS = 0.00821

# Drop /motor_rpm samples this old (seconds). Sustained RPM staleness
# means the bridge stopped publishing; fall back to no velocity prior
# rather than constraining the optimizer with last-good-but-stale data.
RPM_STALE_S = 0.5


class State(Enum):
    INIT_WAITING_IMU = 1
    INIT_CALIBRATING = 2
    SLAM_RUNNING = 3


class ConeGraphSlamNode(BaseLifecycleNode):
    """Cone-graph SLAM as a managed LifecycleNode.

    Lifecycle layout:

      on_configure  declare parameters (re-readable on every
                    re-configure), instantiate components (preint /
                    graph / db), open the optional landmark capture
                    file, create lifecycle publishers + TF
                    broadcasters, emit the map→odom static identity.
      on_activate   reset state machine to INIT_WAITING_IMU, create
                    the four subscriptions (/imu, /Conos_raw,
                    /motor_rpm, /testing_only/odom), super().on_activate
                    flips lifecycle pubs to emitting state.
      on_deactivate destroy subscriptions; pubs go quiet via super().
      on_cleanup    destroy publishers + broadcasters, close the
                    capture file, drop component refs.
    """

    NODE_NAME = "slam_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # --- Parameter declarations live in __init__ so they exist
        # before configure (mode_manager will eventually pre-set the
        # mission strategy flag via parameters).
        # Frames (REP-105). Post-#382 the SLAM-owned absolute frame
        # is `map`, not `odom` — the `odom` frame is supervisor's
        # dead-reckoning, slam computes `map → odom` to absorb the
        # difference. Backward-compat parameter names preserved.
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # Pose-jump sanity check thresholds (#273 follow-up).
        # Maximum allowed deviation between iSAM2's optimized pose and
        # the IMU-predicted pose at each scan. When a wrong cone match
        # passes the DA gate, iSAM2 snaps pose to fit the bad factor;
        # the sanity check catches that and re-anchors at the IMU
        # prediction.
        # 0.8 m: bigger than any legitimate single-scan iSAM2 correction
        #         (typical IMU drift over 100 ms is sub-decimeter, the
        #         correction iSAM2 applies via cone factors is at most
        #         a few centimeters per scan once the map has matured).
        # 0.3 rad (~17°): ditto for yaw — much more than any legitimate
        #         single-scan refinement.
        self.declare_parameter("pose_jump_max_pos_m", 0.8)
        self.declare_parameter("pose_jump_max_yaw_rad", 0.3)

        # Proximity veto for new-landmark *creation* (separate from the
        # DA match gate). When pose drift creeps to ~DA_GATE_M (1.0 m),
        # re-observations of existing cones fall just past the gate,
        # DA flags them as new, and they get spawned as ghost landmarks
        # anchored at the drifted pose — the cascade trigger documented
        # in bag lap_attempt_20260511_142317 (5 of 6 "new" obs were
        # 1.02-1.20 m from an existing landmark; pose had drifted ~1 m
        # off truth). The veto refuses to create a new landmark whose
        # world position is closer than this to any existing landmark.
        # Genuine new cones (track-min spacing ≥ 3 m, gate-min ≥ 1.5 m
        # in FSD rules) still get spawned; cascade ghosts don't.
        # Set to 0 to disable the veto.
        self.declare_parameter("new_landmark_proximity_veto_m", 1.5)

        # Count-based DA-spike threshold (issue #447, post-Phase-2 finding).
        # The original spike detector compared n_new_pre against
        # 60 % of total observations — but a 5-of-14 burst (36 %)
        # slipped through it in lap_postveto_20260511_183638 and
        # lap_ekf_inloop_20260511_235510, spawning the ghosts that
        # detonated the cascade in both runs. Three or more new
        # landmark candidates from a single mature-phase scan is
        # *always* anomalous in FSD: legitimate new-cone arrivals
        # come at vehicle-speed × scan-period spacing (typically 1
        # cone every 3–4 scans), never bursting. Setting this to 0
        # disables the count-based gate; the percentage gate stays
        # active either way.
        self.declare_parameter("cascade_spike_new_count_threshold", 3)

        # Minimum re-observation count before a landmark is published
        # on /Conos for downstream consumers. Single-shot detections
        # are noisy (one-frame perception artifacts, phantom DA spawns
        # from /odom yaw drift) and pollute the cone map the planner
        # sees. Counting how many factors point at a landmark before
        # exposing it filters those out without affecting the SLAM
        # solve itself (the landmark stays in the factor graph, just
        # not on /Conos). 3 is the smallest count that survives
        # genuine perception noise (cone seen → missed → seen again);
        # 0 disables the gate (legacy behaviour).
        self.declare_parameter("min_observations_for_publish", 3)

        # Cascade-skip recovery. After this many consecutive scans
        # rejected by the cascade-spike gate, force-accept the next
        # scan instead of skipping it. The theory: 10 scans in a row
        # all flagged as "all-new" is far longer than any real cascade
        # burst — it's almost certainly the car driving into a section
        # of track with legitimately-new cones that haven't been
        # mapped yet. Refusing to spawn them indefinitely strands
        # SLAM (cones lose alignment with the IMU-only pose dead-
        # reckon, /slam/pose freezes, controller drives blind).
        #
        # The proximity veto (`new_landmark_proximity_veto_m`) still
        # gates individual cones inside the force-accepted scan, so
        # cascade ghosts that are spatially adjacent to existing
        # landmarks are still rejected — we only relax the "skip the
        # whole scan" decision. Set to 0 to disable recovery; the
        # legacy "stay stuck forever" behaviour returns.
        #
        # Tuning: 5 scans @ 10 Hz = 0.5 s of stuck. Originally 10 but
        # bag autocross_track_20260404_013721_20260518_095215 had a
        # cascade that lasted only 9 scans before the obs count dropped
        # to 1 — recovery never fired despite SLAM being structurally
        # in trouble. 5 catches shorter cascades without false-firing
        # on transient ~0.2-0.3 s DA bursts.
        self.declare_parameter("cascade_skip_recovery_threshold", 5)

        # I/O references — populated in on_configure / on_activate.
        self.map_frame: str = ""
        self.odom_frame: str = ""
        self.base_frame: str = ""
        self._preint: Optional[ImuPreintegrator] = None
        self._graph: Optional[FactorGraph] = None
        self._db: Optional[LandmarkDb] = None
        self._latest_result: Optional[ScanResult] = None
        self._state = State.INIT_WAITING_IMU
        self._calib_started_t: Optional[float] = None
        # Count of consecutive scans rejected by the cascade-spike
        # gate. Reset to 0 on any scan that passes through normally;
        # crossing the recovery threshold force-accepts the next scan.
        # See `cascade_skip_recovery_threshold` param.
        self._consecutive_cascade_skips: int = 0

        self._lm_capture_path: str = ""
        self._lm_capture_fh = None

        self._obs_diag: dict = {}
        self._obs_n_scans = 0
        self._obs_last_log_ns = 0

        import time as _time
        self._time = _time
        self._latest_rpm: Optional[float] = None
        self._latest_rpm_t: Optional[float] = None
        self._latest_gt: Optional[Odometry] = None
        self._gt_init_pose: Optional[gtsam.Pose3] = None

        # Subscription handles (created in on_activate)
        self._sub_imu = None
        self._sub_cones = None
        self._sub_rpm = None
        self._sub_gt = None

        # Subscription for sim_supervisor's /odom (Phase 2 #382 —
        # needed to compute map→odom drift correction). Cached
        # latest sample is consumed in _publish_map_to_odom on each
        # scan tick.
        self._sub_supervisor_odom = None
        self._latest_supervisor_odom: Optional[Odometry] = None
        # /odom pose at the previous SCAN tick (not the previous /odom
        # message). Used to compute the EKF's per-scan delta-pose and
        # stage it as a BetweenFactor on X(k-1)→X(k). Reset on activate.
        self._prev_scan_odom_pose: Optional[gtsam.Pose3] = None

        # Publisher / broadcaster handles (created in on_configure)
        self._tf_broadcaster = None
        self._state_pub = None
        self._cones_pub = None
        self._gt_aligned_pub = None
        self._gt_error_pub = None
        # /slam/finished publisher (#384). Always emits default-false
        # on activate; will go true on real mission-completion detection
        # in a follow-up. Stays latched so a late mission_control
        # subscriber inherits the current value.
        self._finished_pub = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        ret = super().on_configure(state)
        if ret != TransitionCallbackReturn.SUCCESS:
            return ret
        self.get_logger().info("on_configure: components + publishers + static TF")

        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        # Components
        self._preint = ImuPreintegrator()
        self._graph = FactorGraph()
        self._db = LandmarkDb()
        self._latest_result = None
        self._state = State.INIT_WAITING_IMU
        self._calib_started_t = None
        self._latest_rpm = None
        self._latest_rpm_t = None
        self._latest_gt = None
        self._gt_init_pose = None
        self._latest_supervisor_odom = None
        self._prev_scan_odom_pose = None

        # Optional per-landmark creation diagnostic. When DV_SLAM_LANDMARK_CAPTURE
        # is set to a writable path, every new landmark dumps one JSON line
        # (id, body_x, body_y, range, color, pred_yaw_deg, step) at creation.
        # Used to verify the colour-lock-on-first-observation hypothesis (#188):
        # if many landmarks created at long range or during a yaw rotation
        # have body_y near the threshold and get locked the wrong colour, the
        # hypothesis is confirmed.
        import os as _os
        self._lm_capture_path = _os.environ.get("DV_SLAM_LANDMARK_CAPTURE", "")
        if self._lm_capture_path:
            try:
                self._lm_capture_fh = open(self._lm_capture_path, "w")
                self.get_logger().info(
                    f"DV_SLAM_LANDMARK_CAPTURE → {self._lm_capture_path}")
            except OSError as ex:
                self.get_logger().error(f"landmark capture open failed: {ex}")
                self._lm_capture_fh = None

        # Publishers (lifecycle — silent until on_activate).
        # Phase 2 (#382): /cone_slam/state renamed to /slam/pose so
        # the topic doesn't bind to "cone_slam" as the algorithm
        # implementation. Frame_id flipped from odom → map (pose is
        # SLAM-absolute, drift-corrected; map→odom→base_link chain
        # resolves to the same value via TF).
        self._state_pub = self.create_lifecycle_publisher(
            Odometry, "/slam/pose", 10)
        self._cones_pub = self.create_lifecycle_publisher(
            MarkerArray, "/Conos", 10)
        # GT-aligned diagnostic odometry. Same Odometry shape as
        # /slam/pose but containing the ground truth re-expressed in
        # SLAM's anchored body-frame world. SLAM-vs-GT divergence in
        # this frame is the actual SLAM drift; in the raw GT frame it
        # would be that drift plus the static frame mismatch. Topic
        # name stays under /cone_slam/* — that's the diagnostic-only
        # namespace, separate from /slam/* which is the production
        # consumer surface.
        self._gt_aligned_pub = self.create_lifecycle_publisher(
            Odometry, "/cone_slam/gt_aligned", 10)
        self._gt_error_pub = self.create_lifecycle_publisher(
            Float32, "/cone_slam/gt_error_m", 10)

        # /slam/finished — mission-completion signal consumed by
        # mission_control_node (#384). Latched + default-false so a
        # late-joining mission_control sees an unambiguous starting
        # state. Pre-#384 slam had no way to signal mission-end
        # directly (control_node's stop-latch handled it via braking
        # to zero); post-#384 mission_control needs an explicit
        # rising-edge signal to close the RuntimeControl action with
        # outcome="finished". Currently a stub — slam still uses its
        # internal stop-anchor logic in control_node for braking;
        # this publisher just emits the default false until a future
        # PR wires the lap-min-distance + big-orange detector to it.
        finished_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._finished_pub = self.create_lifecycle_publisher(
            Bool, "/slam/finished", finished_qos)

        # TF broadcaster — non-lifecycle (tf2 doesn't ship lifecycle
        # variants). Used to publish the dynamic `map → odom` drift
        # correction inside _publish_map_to_odom, called per scan
        # tick from _on_cones. Pre-#382 slam owned odom→base_link
        # AND a static map→odom identity; both are retired.
        # sim_supervisor now owns odom→base_link at 100 Hz; slam owns
        # map→odom at scan rate (~10 Hz).
        self._tf_broadcaster = TransformBroadcaster(self)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info(
            "on_activate: subscriptions + state-machine reset "
            f"(map='{self.map_frame}', odom='{self.odom_frame}', "
            f"base='{self.base_frame}')")

        # Reset state machine and runtime state. A deactivate→activate
        # cycle should look like a fresh run, not a resumed one — the
        # IMU preintegrator can't bridge the deactivated gap, and any
        # downstream consumer reading /tf during the gap will already
        # have stale data.
        self._state = State.INIT_WAITING_IMU
        self._calib_started_t = None
        self._latest_result = None
        self._preint = ImuPreintegrator()
        self._graph = FactorGraph()
        self._db = LandmarkDb()
        self._latest_rpm = None
        self._latest_rpm_t = None
        self._latest_gt = None
        self._gt_init_pose = None
        self._latest_supervisor_odom = None
        self._prev_scan_odom_pose = None
        self._obs_diag = {}
        self._obs_n_scans = 0
        self._obs_last_log_ns = 0
        self._consecutive_cascade_skips = 0

        # Subscriptions.
        # Bigger IMU queue than qos_profile_sensor_data's default 10:
        # at 400 Hz we get ~40 samples per 100 ms scan window, and
        # rclpy's single-threaded executor cannot service the IMU
        # callback while the cone callback's iSAM2 update is running
        # (~tens of ms). With queue=10 we drop most of the window's
        # samples and the preintegrator sees ~3-5 instead of ~40.
        # 2000 leaves headroom for several scan windows of buffer.
        imu_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2000,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_imu = self.create_subscription(
            Imu, "/imu", self._on_imu, imu_qos)
        # /Conos_raw is the scan trigger AND the source of cone
        # observations. Cone_Detection publishes ~10 Hz, in base_link
        # frame, with cone height encoded on marker.scale.z.
        self._sub_cones = self.create_subscription(
            MarkerArray, "/Conos_raw", self._on_cones, 10)

        # /motor_rpm — bridge publishes at 100 Hz from getCarState().
        # We don't fire on every sample; we cache the latest and consume
        # it inside the scan callback as a velocity prior. BEST_EFFORT
        # is fine because the cache holds the last value through any
        # single dropped sample.
        rpm_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_rpm = self.create_subscription(
            Float32, "/motor_rpm", self._on_rpm, rpm_qos)

        # /testing_only/odom — sim ground truth, used only for the
        # GT-aligned diagnostic. The bridge encodes this in ENU
        # (East-North-Up); SLAM internally anchors to the car's initial
        # pose, so direct comparison would mix two different world
        # frames. We snapshot the GT pose at calibration-end and publish
        # a pose-aligned residual on /cone_slam/gt_aligned. Bridge
        # publishes BEST_EFFORT (per `sensor_qos` in
        # ifssim_ros_wrapper.cpp:289); subscriber must match or messages
        # silently never arrive.
        gt_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_gt = self.create_subscription(
            Odometry, "/testing_only/odom", self._on_gt_odom, gt_qos)

        # /odom from sim_supervisor (Phase 2 — #382). Cached and
        # consumed in _publish_map_to_odom on every scan tick to
        # compute the drift-correction transform. RELIABLE since
        # supervisor publishes at 100 Hz and we only need fresh
        # samples every ~100 ms — dropped samples would cause
        # transient map→odom inconsistency.
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_supervisor_odom = self.create_subscription(
            Odometry, "/odom", self._on_supervisor_odom, odom_qos)

        # Latch /slam/finished=false on activate (#384 stub). The
        # publisher exists from on_configure; this fires the default
        # value so a subscriber that joins between configure and
        # activate sees a defined latched state.
        if self._finished_pub is not None:
            self._finished_pub.publish(Bool(data=False))

        return super().on_activate(state)

    def on_deactivate(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate: dropping subscriptions")
        for sub in (self._sub_imu, self._sub_cones,
                    self._sub_rpm, self._sub_gt,
                    self._sub_supervisor_odom):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_imu = None
        self._sub_cones = None
        self._sub_rpm = None
        self._sub_gt = None
        self._sub_supervisor_odom = None
        return super().on_deactivate(state)

    def on_cleanup(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup: destroying publishers + components")
        for sub in (self._sub_imu, self._sub_cones,
                    self._sub_rpm, self._sub_gt,
                    self._sub_supervisor_odom):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_imu = None
        self._sub_cones = None
        self._sub_rpm = None
        self._sub_gt = None
        self._sub_supervisor_odom = None

        for pub in (self._state_pub, self._cones_pub,
                    self._gt_aligned_pub, self._gt_error_pub,
                    self._finished_pub):
            if pub is not None:
                self.destroy_publisher(pub)
        self._state_pub = None
        self._cones_pub = None
        self._gt_aligned_pub = None
        self._gt_error_pub = None
        self._finished_pub = None

        # tf2 broadcasters are not lifecycle-aware; drop the ref.
        # The static map→odom broadcaster was retired in #382 (map→odom
        # is now dynamic, computed from slam_pose ⊖ /odom).
        self._tf_broadcaster = None
        self._latest_supervisor_odom = None
        self._prev_scan_odom_pose = None

        # Components
        self._preint = None
        self._graph = None
        self._db = None
        self._latest_result = None

        if self._lm_capture_fh is not None:
            try:
                self._lm_capture_fh.close()
            except Exception:
                pass
            self._lm_capture_fh = None

        return super().on_cleanup(state)

    def on_shutdown(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        # Reuse cleanup logic — shutdown after cleanup is a no-op.
        if self._lm_capture_fh is not None:
            try:
                self._lm_capture_fh.close()
            except Exception:
                pass
            self._lm_capture_fh = None
        return TransitionCallbackReturn.SUCCESS

    # ----- RPM callback ------------------------------------------------------

    def _on_rpm(self, msg: Float32) -> None:
        # Convert motor RPM → body-frame longitudinal velocity (m/s).
        # We cache instead of firing optimizer steps off RPM because RPM
        # arrives at 100 Hz and scans arrive at 10 Hz; one prior per
        # scan is what the factor graph wants.
        self._latest_rpm = float(msg.data) * RPM_TO_MS
        self._latest_rpm_t = self._time.monotonic()

    def _on_gt_odom(self, msg: Odometry) -> None:
        # Cache the latest /testing_only/odom sample. Used only to
        # publish the SLAM-vs-GT residual on /cone_slam/gt_aligned —
        # never feeds into the factor graph (that would defeat the
        # purpose of a SLAM diagnostic).
        self._latest_gt = msg
        # Lazy alignment anchor: if SLAM has already transitioned to
        # SLAM_RUNNING but no GT sample had arrived in time for
        # _finish_calibration to snapshot one, take the first sample
        # that does arrive as the anchor. Off by < 100 ms typically;
        # immaterial for a SLAM-drift visualisation.
        if (self._state == State.SLAM_RUNNING
                and self._gt_init_pose is None):
            self._gt_init_pose = _odom_to_pose3(msg)
            self.get_logger().info(
                f"GT alignment anchor (late): "
                f"pos=({self._gt_init_pose.x():+.2f}, {self._gt_init_pose.y():+.2f}), "
                f"yaw={np.degrees(self._gt_init_pose.rotation().yaw()):+.1f}°"
            )

    # ----- IMU callback ------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])
        self._preint.push_sample(ImuSample(t=t, accel=accel, gyro=gyro))

        if self._state == State.INIT_WAITING_IMU:
            self._calib_started_t = t
            self._state = State.INIT_CALIBRATING
            self.get_logger().info(
                "first IMU received — INIT_CALIBRATING (stationary, "
                f"{CALIBRATION_SECONDS:.1f} s)")

        elif self._state == State.INIT_CALIBRATING:
            if self._preint.has_enough_for_calibration(CALIBRATION_SECONDS):
                self._finish_calibration()

    def _finish_calibration(self) -> None:
        accel_bias, gyro_bias, gravity_body = self._preint.estimate_bias()
        self.get_logger().info(
            "calibration done — "
            f"accel_bias=({accel_bias[0]:+.4f},{accel_bias[1]:+.4f},"
            f"{accel_bias[2]:+.4f}) m/s², "
            f"gyro_bias=({gyro_bias[0]:+.5f},{gyro_bias[1]:+.5f},"
            f"{gyro_bias[2]:+.5f}) rad/s, "
            f"gravity_body=({gravity_body[0]:+.3f},{gravity_body[1]:+.3f},"
            f"{gravity_body[2]:+.3f}) m/s²")

        # Anchor x_0 at world origin, stationary.
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        self._graph.initialize_anchor(
            initial_pose=gtsam.Pose3(),
            initial_velocity=np.zeros(3),
            initial_bias=bias,
        )
        self._latest_result = self._graph.latest()
        self._state = State.SLAM_RUNNING
        self.get_logger().info("SLAM_RUNNING — pose graph anchored at origin")

        # Snapshot the GT pose at this instant. SLAM's pose at t=0 is
        # identity (pose graph anchored at origin); GT is at whatever
        # (ENU) world coordinates UE5 spawned the car at. Future SLAM
        # poses are in a frame whose origin == car spawn, x = car-initial
        # forward, y = car-initial left. To compare to GT we have to
        # subtract this snapshot and rotate by -initial_yaw_enu.
        if self._latest_gt is not None:
            self._gt_init_pose = _odom_to_pose3(self._latest_gt)
            self.get_logger().info(
                f"GT alignment anchor: "
                f"pos=({self._gt_init_pose.x():+.2f}, {self._gt_init_pose.y():+.2f}), "
                f"yaw={np.degrees(self._gt_init_pose.rotation().yaw()):+.1f}°"
            )
        else:
            self.get_logger().warn(
                "no /testing_only/odom received before SLAM_RUNNING — "
                "GT-aligned diagnostic disabled this run")

    # ----- Cone observation callback (scan trigger) -------------------------

    def _on_cones(self, msg: MarkerArray) -> None:
        if self._state != State.SLAM_RUNNING:
            return
        if self._latest_result is None:
            return

        # MarkerArray has no top-level header, but every Cone_Detection
        # marker carries the originating LiDAR scan's header.stamp.
        stamp = self._stamp_msg(msg)
        if stamp is None:
            # Empty array — nothing to do.
            return
        t_scan = stamp.sec + stamp.nanosec * 1e-9

        # Stage IMU preintegration for this scan window.
        try:
            pim, _dt = self._preint.integrate_to(t_scan)
        except RuntimeError as e:
            self.get_logger().warn(f"skip scan: {e}")
            return
        self._graph.stage_imu_factor(pim, self._latest_result)

        # Stage the motor-RPM velocity prior on V(k). This is the
        # anchor that prevents the optimizer from rotating the global
        # frame to find a cheaper minimum during cone-poor windows
        # (the cascade root cause). Skipped if RPM is stale or hasn't
        # arrived yet — the prior would be misleading rather than
        # helpful in that regime.
        if self._latest_rpm is not None and self._latest_rpm_t is not None:
            age = self._time.monotonic() - self._latest_rpm_t
            if age <= RPM_STALE_S:
                # Use the same predicted yaw we'll use for DA below.
                _nav_state = gtsam.NavState(
                    self._latest_result.pose, self._latest_result.velocity)
                _pred_yaw = pim.predict(
                    _nav_state, self._latest_result.bias).pose().rotation().yaw()
                self._graph.stage_velocity_prior(
                    v_body_long=self._latest_rpm,
                    predicted_yaw=_pred_yaw,
                )

        # Stage the /odom-derived BetweenFactor (the "trust the EKF"
        # channel into the graph). /odom is the post-#534/#539 EKF's
        # fused IMU + RPM + steering estimate with Coriolis correction.
        # During cone-poor windows (the cascade-spike root cause), this
        # factor anchors X(k) to a quality estimate that's drift-tracked
        # by all available sensors — much better than SLAM's IMU-only
        # internal fallback which produced 10 m / 94° pose jumps during
        # cornering on bag _095215. First scan after activate has no
        # previous cached /odom pose; we just record this one and skip
        # the factor — the IMU+RPM priors carry the load that scan.
        if self._latest_supervisor_odom is not None:
            cur_odom_pose = _odom_to_pose3(self._latest_supervisor_odom)
            if self._prev_scan_odom_pose is not None:
                between_pose = (self._prev_scan_odom_pose.inverse()
                                .compose(cur_odom_pose))
                self._graph.stage_odom_between(between_pose)
            self._prev_scan_odom_pose = cur_odom_pose

        # Parse cone observations from /Conos_raw markers.
        observations = self._observations_from_markers(msg)

        # Per-scan obs tally for the SLAM_OBS diagnostic. No colour
        # breakdown anymore — the body_y classifier is gone, every
        # cone is just a position.
        per_scan = {"obs_total": len(observations)}

        # Run data association against the predicted body-frame position
        # of every existing landmark, using the iSAM2-predicted pose
        # from the IMU step (still inside the optimizer's "predicted"
        # state — we've staged X(k) but not committed yet, so use the
        # pose that pim.predict produced from the previous result).
        nav_state = gtsam.NavState(
            self._latest_result.pose, self._latest_result.velocity)
        predicted_pose = pim.predict(
            nav_state, self._latest_result.bias).pose()
        pred_x = predicted_pose.x()
        pred_y = predicted_pose.y()
        pred_yaw = predicted_pose.rotation().yaw()

        # Mahalanobis DA stays disabled. Three variants tested on
        # 2026-04-29:
        # (1) full pose-aware Mahalanobis (4×/16× covariance inflation
        #     + 0.49 m² floor): cascaded at t≈75s. iSAM2 marginal is
        #     internal certainty not actual error; even with inflation
        #     the gate is wrong during empty-scan-driven pose drift.
        # (2) Mahalanobis gate + Euclidean Hungarian cost: same.
        # (3) Landmark-cov-only Mahalanobis (no pose Jacobian): same.
        # In every variant, mid-drive tracking was comparable to
        # Euclidean (60 s ≈ 0.8 m) but the cascade still triggered
        # at the same lap position because the cascade root cause is
        # pose drift > gate during empty-scan windows — no DA
        # strategy can fix this because there's nothing to associate.
        # Real fix needs lost-track detection / scan rejection during
        # pose-prediction-confidence collapse, not gate widening.
        # APIs in factor_graph (pose_covariance, landmark_covariance)
        # and data_association (inflation constants) stay in place
        # for future revisits.
        # Pass current_step so associate() can expand per-landmark
        # gates for landmarks that haven't been associated recently —
        # the recovery mechanism for the rejection bursts triggered
        # by improvement A.
        matches = associate(
            observations, pred_x, pred_y, pred_yaw, self._db,
            current_step=self._graph.step)

        # Pre-stage cascade-trigger detection. The cascade signature
        # observed on trackA_manual_001602 around t≈80 s is: a single
        # scan flips DA from "steady, mostly-associated" to "mostly
        # new" (e.g., obs=8 new=6 assoc=2). The optimizer then jumps
        # pose to accommodate the falsely-new landmarks and the graph
        # never recovers. Detection: if the new-rate suddenly spikes
        # when (a) we have ≥5 observations to be statistically
        # meaningful, (b) we're past the early-discovery phase
        # (step > 30, so most cones in the local map are mature),
        # (c) >60 % of obs are flagged new — the predicted pose is
        # likely wrong and committing the cone factors would corrupt
        # the graph.
        #
        # Recovery (#273): commit the IMU factor only, skip the cone
        # factors. Earlier behaviour discarded EVERYTHING (the IMU
        # factor too) and returned, freezing the graph at the prev
        # committed pose. While the car physically moved during the
        # skipped scans, the predicted pose for the next scan stayed
        # stale — so when DA recovered, observed cones were all far
        # from their landmarks (even further apart than the real drift
        # the IMU would have indicated), producing yet more "all-new"
        # scans, more skips, more drift. By committing the IMU we keep
        # pose dead-reckoning during the skipped window; drift over a
        # few hundred ms of IMU-only update is much smaller than over
        # the same window of pose-freeze.
        n_new_pre  = sum(1 for m in matches if m.landmark_id == -1)
        total_pre  = len(matches)
        # Two complementary triggers, both gated on step > 30
        # (post-discovery — the local cone map is mature, every
        # observation should be re-association of a known landmark
        # with high probability):
        #
        #   A. **Percentage trigger** (legacy): >60 % of the scan's
        #      observations flagged new. Catches "everything looks new"
        #      cascades — observed pre-#441 on bags with extreme drift.
        #      Requires ≥5 obs to be statistically meaningful.
        #
        #   B. **Count trigger** (post-#441 finding from in-loop runs):
        #      ≥ N new in a single scan, where N is small. Catches the
        #      5-of-14 (36 %) and 8-of-13 (62 %) bursts that slipped
        #      through the percentage gate in lap_postveto_20260511_142317
        #      and lap_ekf_inloop_20260511_235510. FSD cone spacing
        #      means legitimate new arrivals come at most 1 per ~3
        #      scans at typical speeds — anything ≥ N=3 in one scan
        #      is anomalous. Default 3, tunable via parameter, 0
        #      disables.
        n_count_thresh = int(self.get_parameter(
            "cascade_spike_new_count_threshold").value)
        triggered, reasons = cascade_spike_triggered(
            n_new_pre, total_pre, self._graph.step,
            count_threshold=n_count_thresh)

        # Cascade-skip recovery: if we've been skipping continuously
        # for too long, force-accept this scan. Long-stretch skips
        # are almost certainly new-territory exploration (legitimate
        # cones not yet in the map), not an adversarial cascade.
        # Refusing forever strands SLAM in IMU-only dead reckoning
        # and the /slam/pose feed freezes for the controller.
        if triggered:
            self._consecutive_cascade_skips += 1
            recovery_thresh = int(self.get_parameter(
                "cascade_skip_recovery_threshold").value)
            if (recovery_thresh > 0
                    and self._consecutive_cascade_skips > recovery_thresh):
                self.get_logger().warn(
                    f"CASCADE_SKIP_RECOVERY: force-accepting scan "
                    f"after {self._consecutive_cascade_skips} "
                    f"consecutive skips — assuming new-territory "
                    f"exploration (obs={total_pre} new={n_new_pre} "
                    f"assoc={total_pre - n_new_pre}); proximity veto "
                    f"still gates individual cones."
                )
                self._consecutive_cascade_skips = 0
                triggered = False  # fall through to the spawn path

        if triggered:
            self.get_logger().warn(
                f"skip cone factors: DA-failure spike "
                f"[{', '.join(reasons)}] "
                f"(obs={total_pre} new={n_new_pre} "
                f"assoc={total_pre - n_new_pre}) — IMU-only update")
            # Commit the staged IMU factor + bias-RW factor (the only
            # things staged at this point — cone factors are staged
            # only after this check). iSAM2 advances pose by IMU
            # prediction; no cone constraints applied this scan.
            result = self._graph.commit()
            self._latest_result = result
            self._preint.update_bias(result.bias)
            self._db.update_from_estimate(self._graph.landmark_position)
            self._publish_map_to_odom(stamp, result)
            self._publish_state(stamp, result)
            self._publish_cone_map(stamp)
            per_scan["skipped"] = 1
            self._accumulate_obs_diag(per_scan)
            return
        else:
            # Scan passed the gate cleanly — reset the consecutive-skip
            # counter so a fresh stretch of skips later can be detected.
            self._consecutive_cascade_skips = 0

        # For each matched obs → factor between current pose and the
        # known landmark. For unmatched → allocate a new landmark and
        # add a factor to it. Before allocation, apply the proximity
        # veto: refuse to spawn a landmark within
        # `new_landmark_proximity_veto_m` of any existing one (cascade
        # guard — see parameter docstring).
        veto_m = float(self.get_parameter(
            "new_landmark_proximity_veto_m").value)
        n_new = 0
        n_assoc = 0
        n_vetoed = 0
        for o, m in zip(observations, matches):
            if m.landmark_id == -1:
                world_xyz = self._body_to_world(
                    o.body_x, o.body_y, pred_x, pred_y, pred_yaw)
                if veto_m > 0.0:
                    d_near = self._db.nearest_xy_distance_m(world_xyz)
                    if d_near < veto_m:
                        n_vetoed += 1
                        continue
                lm = self._db.create(world_xyz, self._graph.step)
                self._graph.stage_new_landmark(lm.id, world_xyz)
                self._graph.stage_cone_observation(
                    lm.id, o.body_x, o.body_y, o.sigma_xy)
                n_new += 1
                if self._lm_capture_fh is not None:
                    try:
                        import math as _math, json as _json
                        rng = _math.hypot(o.body_x, o.body_y)
                        bearing_deg = _math.degrees(
                            _math.atan2(o.body_y, o.body_x))
                        self._lm_capture_fh.write(_json.dumps({
                            "id": lm.id,
                            "step": self._graph.step,
                            "body_x": o.body_x,
                            "body_y": o.body_y,
                            "range_m": rng,
                            "bearing_deg": bearing_deg,
                            "height": o.height,
                            "pose_yaw_deg": _math.degrees(pred_yaw),
                            "world_xyz": [float(world_xyz[0]),
                                          float(world_xyz[1]),
                                          float(world_xyz[2])],
                        }) + "\n")
                        self._lm_capture_fh.flush()
                    except Exception:
                        pass
            else:
                self._db.mark_observed(m.landmark_id, self._graph.step)
                self._graph.stage_cone_observation(
                    m.landmark_id, o.body_x, o.body_y, o.sigma_xy)
                n_assoc += 1
        per_scan["new"] = n_new
        per_scan["assoc"] = n_assoc
        per_scan["vetoed"] = n_vetoed
        if n_vetoed > 0:
            self.get_logger().info(
                f"proximity-veto: dropped {n_vetoed} would-be-new "
                f"landmark(s) within {veto_m:.2f} m of existing ones "
                f"(cascade guard; obs={len(observations)} "
                f"new={n_new} assoc={n_assoc})")

        self._accumulate_obs_diag(per_scan)

        # Commit IMU + cone factors with a post-commit pose-jump
        # sanity check (#273 follow-up). The cascade detector above
        # catches *symptoms* — bursts of all-NEW observations — but
        # only AFTER a bad cone match has already snapped pose. The
        # sanity check below catches the *cause*: an iSAM2 update
        # that pushes pose far from where IMU prediction says we are.
        # When it fires, a strong prior at the IMU-predicted pose is
        # added and the graph is re-optimized; pose at this step lands
        # near IMU prediction instead of where the bad cone factor
        # tried to drag it.
        # Gate the sanity check the same way the cascade detector is
        # gated: only fire after the early-discovery phase
        # (`step > 30`). Reason: iSAM2 refines the IMU bias estimate
        # over the first ~30 scans, during which the optimized pose
        # legitimately deviates from the IMU prediction by tens of cm
        # as it incorporates the first cone constraints. Triggering
        # the corrective prior in that window pins the pose to the
        # uncalibrated-bias prediction and prevents iSAM2 from
        # converging.
        max_pos_dev_m = self.get_parameter("pose_jump_max_pos_m").value
        max_yaw_dev_rad = self.get_parameter("pose_jump_max_yaw_rad").value
        if self._graph.step > 30:
            result, was_corrected = self._graph.commit_with_pose_sanity_check(
                predicted_pose, max_pos_dev_m, max_yaw_dev_rad)
            if was_corrected:
                self.get_logger().warn(
                    f"pose-jump rejected: snapped to IMU prediction at "
                    f"step={self._graph.step} "
                    f"(thresholds: {max_pos_dev_m:.2f} m, "
                    f"{np.degrees(max_yaw_dev_rad):.1f}°)")
        else:
            result = self._graph.commit()
        self._latest_result = result
        self._preint.update_bias(result.bias)

        # Refresh the working landmark estimates so the next DA step
        # uses iSAM2-corrected positions, not stale initial guesses.
        self._db.update_from_estimate(self._graph.landmark_position)

        self._publish_map_to_odom(stamp, result)
        self._publish_state(stamp, result)
        self._publish_cone_map(stamp)

        # Quiet log every 10 scans (~1 Hz at 10 Hz LiDAR).
        if self._graph.step % 10 == 0:
            self.get_logger().info(
                f"step={self._graph.step} "
                f"obs={len(observations)} new={n_new} assoc={n_assoc} "
                f"map={len(self._db)} "
                f"pose=({result.pose.x():+.2f},{result.pose.y():+.2f},"
                f"yaw={np.degrees(result.pose.rotation().yaw()):+.1f}°)")

        # Bias trajectory dump every 50 scans (~5 s wall, ~5 % of a lap).
        # Tagged so it greps cleanly out of the SLAM log: "BIAS step=…".
        # Used to diagnose whether iSAM2 is letting the bias drift away
        # from the calibration value over the lap, vs. holding it locked
        # by BIAS_RW_SIGMAS being too tight.
        if self._graph.step % 50 == 0:
            ab = result.bias.accelerometer()
            gb = result.bias.gyroscope()
            v  = result.velocity
            self.get_logger().info(
                f"BIAS step={self._graph.step} "
                f"accel=({ab[0]:+.5f},{ab[1]:+.5f},{ab[2]:+.5f}) m/s² "
                f"gyro=({gb[0]:+.6f},{gb[1]:+.6f},{gb[2]:+.6f}) rad/s "
                f"vel=({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}) m/s "
                f"|v|={float(np.linalg.norm(v)):.3f}")

    # ----- helpers ----------------------------------------------------------

    def _accumulate_obs_diag(self, per_scan: dict) -> None:
        """Accumulate per-scan obs/assoc/new counters and emit a
        per-second SLAM_OBS log line. Compares observations entering
        SLAM with what survives data association."""
        for k, v in per_scan.items():
            self._obs_diag[k] = self._obs_diag.get(k, 0) + v
        self._obs_n_scans += 1
        now_ns = self.get_clock().now().nanoseconds
        if self._obs_last_log_ns == 0:
            self._obs_last_log_ns = now_ns
            return
        if now_ns - self._obs_last_log_ns < 1_000_000_000:
            return
        n = self._obs_n_scans
        if n <= 0:
            return
        d = self._obs_diag
        def _avg(key: str) -> float:
            return d.get(key, 0) / n
        self.get_logger().info(
            f"SLAM_OBS (avg/scan over {n}): "
            f"obs={_avg('obs_total'):4.1f} "
            f"assoc={_avg('assoc'):4.1f} "
            f"new={_avg('new'):3.1f} "
            f"vetoed={_avg('vetoed'):.1f} "
            f"skip={_avg('skipped'):.1f}"
        )
        self._obs_diag = {}
        self._obs_n_scans = 0
        self._obs_last_log_ns = now_ns

    @staticmethod
    def _stamp_msg(msg: MarkerArray):
        """Return the first non-DELETE marker's header.stamp, or None
        if the array is empty / DELETEALL only.

        MarkerArray has no top-level header, but Cone_Detection sets
        every marker's stamp from the originating LiDAR scan, so any
        of them is fine.
        """
        for m in msg.markers:
            if m.action != Marker.DELETEALL:
                return m.header.stamp
        return None

    @staticmethod
    def _observations_from_markers(msg: MarkerArray) -> list[Observation]:
        out: list[Observation] = []
        for m in msg.markers:
            # The first marker in the array is action=DELETEALL with
            # placeholder pose — Cone_Detection uses it to clear stale
            # markers in RViz/Foxglove. Skip it.
            if m.action == Marker.DELETEALL:
                continue
            x = m.pose.position.x
            y = m.pose.position.y
            if (x * x + y * y) > MAX_OBSERVATION_RANGE_M * MAX_OBSERVATION_RANGE_M:
                continue
            height = m.scale.z if m.scale.z > 0 else 0.0
            # Detection encodes per-cone σ_xy (metres) on scale.x; the
            # legacy 0.1 default flags "no σ reported" and SLAM falls
            # back to its range-only formula for backward compat.
            sigma_xy = m.scale.x if (m.scale.x > 0.0 and m.scale.x != 0.1) else -1.0
            out.append(Observation(
                body_x=x, body_y=y, height=height, sigma_xy=sigma_xy,
            ))
        return out

    @staticmethod
    def _body_to_world(
        body_x: float, body_y: float,
        pose_x: float, pose_y: float, pose_yaw: float,
    ) -> np.ndarray:
        """Project a body-frame xy into world-frame xyz (z=0)."""
        c = np.cos(pose_yaw)
        s = np.sin(pose_yaw)
        return np.array([
            pose_x + body_x * c - body_y * s,
            pose_y + body_x * s + body_y * c,
            0.0,
        ])

    # ----- output ------------------------------------------------------------

    def _on_supervisor_odom(self, msg: Odometry) -> None:
        """Cache sim_supervisor's latest /odom sample for map→odom math."""
        self._latest_supervisor_odom = msg

    def _publish_map_to_odom(self, stamp, result: ScanResult) -> None:
        """Broadcast the dynamic `map → odom` transform (Phase 2 #382).

        We need a SE(2) transform that, composed with the supervisor's
        latest `odom → base_link`, yields slam's absolute pose in map:

            T_map_base   = T_map_odom · T_odom_base
            ⇒ T_map_odom = T_map_base · T_odom_base⁻¹

        In 2D this is:
            Δyaw = slam_yaw - odom_yaw
            Δpos = slam_pos - R(Δyaw) · odom_pos

        Fallback: until the supervisor's `/odom` is flowing (the
        first ~3 s after activate, during the filter's stationary
        calibration window), broadcast map → odom as identity. This
        keeps Lichtblick's TF tree rooted; the chain map→odom→base_link
        is fully dynamic post-Phase-2, no /tf_static needed.
        """
        slam_pose = result.pose
        slam_x = slam_pose.x()
        slam_y = slam_pose.y()
        slam_yaw = slam_pose.rotation().yaw()

        if self._latest_supervisor_odom is None:
            # Identity fallback during the supervisor calibration window
            # ^ if supervisor hasn't started, treat its odom frame as
            # coincident with map; map→odom = slam_pose itself, so
            # downstream consumers still see slam's pose at the leaf.
            dx, dy, dyaw = slam_x, slam_y, slam_yaw
        else:
            sup = self._latest_supervisor_odom.pose.pose
            sup_x = sup.position.x
            sup_y = sup.position.y
            # Supervisor's quaternion is axis-z only (2D yaw); the
            # full-precision recovery is 2·atan2(qz, qw). q.w can be
            # negative but yaw stays in (-π, π] from atan2.
            sup_yaw = 2.0 * np.arctan2(sup.orientation.z, sup.orientation.w)
            dx, dy, dyaw = compute_map_to_odom(
                slam_x, slam_y, slam_yaw,
                sup_x, sup_y, sup_yaw,
            )

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = float(dx)
        t.transform.translation.y = float(dy)
        t.transform.translation.z = 0.0
        half = 0.5 * float(dyaw)
        t.transform.rotation.w = float(np.cos(half))
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(np.sin(half))
        self._tf_broadcaster.sendTransform(t)

    def _publish_state(self, stamp, result: ScanResult) -> None:
        msg = Odometry()
        msg.header.stamp = stamp
        # Phase 2 (#382): pose is SLAM's absolute estimate — map frame,
        # not odom. Twist still lives in child_frame_id (base_link).
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.base_frame
        pose = result.pose
        msg.pose.pose.position.x = pose.x()
        msg.pose.pose.position.y = pose.y()
        msg.pose.pose.position.z = pose.z()
        q = pose.rotation().toQuaternion()
        msg.pose.pose.orientation.w = q.w()
        msg.pose.pose.orientation.x = q.x()
        msg.pose.pose.orientation.y = q.y()
        msg.pose.pose.orientation.z = q.z()
        # nav_msgs/Odometry semantics: twist is expressed in child_frame_id
        # (here base_link), NOT the header frame. GTSAM's NavState carries
        # velocity in the navigation (odom) frame, so we project to body
        # frame before publishing — otherwise consumers like Control read
        # `twist.linear.x` as longitudinal speed and instead get an axis-
        # aligned world component, which is wrong as soon as the car is
        # not pointing along world +X.
        v_world = result.velocity
        c, s = np.cos(pose.rotation().yaw()), np.sin(pose.rotation().yaw())
        # R_w2b = [[ c, s, 0], [-s, c, 0], [0, 0, 1]]; vertical untouched.
        msg.twist.twist.linear.x = float(c * v_world[0] + s * v_world[1])
        msg.twist.twist.linear.y = float(-s * v_world[0] + c * v_world[1])
        msg.twist.twist.linear.z = float(v_world[2])
        self._state_pub.publish(msg)

        # GT-aligned diagnostic: publish where the ground truth says
        # the car is, expressed in SLAM's anchored body-frame world,
        # plus the position-error magnitude. Both are zero at t=0 by
        # construction; non-zero values are real SLAM drift, not a
        # frame-mismatch artifact.
        self._publish_gt_aligned(stamp, msg)

    def _publish_gt_aligned(self, stamp, slam_msg: Odometry) -> None:
        """Re-express ground truth in SLAM's anchored frame and publish.

        SLAM internally anchors at identity, so the SLAM frame is:
            origin = car spawn (the ENU pose at calibration end)
            +x axis = car's initial forward direction
            +y axis = car's initial left direction
        The bridge publishes /testing_only/odom in ENU. To compare like-
        for-like, we subtract the snapshot pose from each GT sample and
        rotate by -initial_yaw_enu.
        """
        if self._gt_init_pose is None or self._latest_gt is None:
            return
        gt_now = _odom_to_pose3(self._latest_gt)
        # gt_in_slam_frame = init⁻¹ · gt_now
        gt_aligned = self._gt_init_pose.inverse().compose(gt_now)

        out = Odometry()
        out.header.stamp = stamp
        # GT-aligned diagnostic lives in the same frame as /slam/pose
        # so Lichtblick can plot them on the same axis post-Phase-2.
        out.header.frame_id = self.map_frame
        out.child_frame_id = "gt"
        out.pose.pose.position.x = gt_aligned.x()
        out.pose.pose.position.y = gt_aligned.y()
        out.pose.pose.position.z = gt_aligned.z()
        q = gt_aligned.rotation().toQuaternion()
        out.pose.pose.orientation.w = q.w()
        out.pose.pose.orientation.x = q.x()
        out.pose.pose.orientation.y = q.y()
        out.pose.pose.orientation.z = q.z()
        self._gt_aligned_pub.publish(out)

        # Position-error magnitude in metres. Plot this on a second
        # axis to track drift growth without having to do the
        # subtraction in the visualiser.
        dx = slam_msg.pose.pose.position.x - gt_aligned.x()
        dy = slam_msg.pose.pose.position.y - gt_aligned.y()
        err = float(np.hypot(dx, dy))
        err_msg = Float32()
        err_msg.data = err
        self._gt_error_pub.publish(err_msg)

    def _publish_cone_map(self, stamp) -> None:
        """Publish the persistent cone landmark database to /Conos.

        Filters out single-shot / under-observed landmarks. A cone
        whose factor count is < `min_observations_for_publish` is
        kept in the SLAM solve (still in self._db, still a factor in
        the graph) but withheld from /Conos so the planner never
        navigates around phantoms. Cf. autocross_track_20260404_
        013721_20260517_232000: /Conos grew 13 → 130 over 68 s; a
        real autocross has ~60–100 cones, so ~30–40 of those are
        single-shot ghosts from imperfect DA accumulating against
        /odom yaw drift.

        marker.id encodes the persistent SLAM landmark id so downstream
        consumers (path_planning) can track cone identity across scans.
        """
        if len(self._db) == 0:
            return

        min_obs = int(self.get_parameter("min_observations_for_publish").value)

        out = MarkerArray()
        # First marker is DELETEALL so visualizers refresh cleanly.
        # Phase 2 (#382): /Conos now lives in map frame, not odom.
        # Landmark positions in self._db were always SLAM-absolute;
        # only the frame label needed updating.
        delete_all = Marker()
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = self.map_frame
        delete_all.action = Marker.DELETEALL
        out.markers.append(delete_all)

        # Single neutral colour for every landmark — SLAM has no
        # per-cone colour anymore. Yellow for visibility on dark
        # backgrounds; the path planner ignores the colour anyway and
        # routes everything through ConeTypes.UNKNOWN (#268). The
        # per-cone marker.id still encodes the persistent landmark id
        # so downstream consumers (path_planning) can identify cones
        # across scans.
        n_published = 0
        n_filtered = 0
        for lm in self._db:
            if lm.n_observations < min_obs:
                n_filtered += 1
                continue
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.map_frame
            m.id = lm.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(lm.position[0])
            m.pose.position.y = float(lm.position[1])
            m.pose.position.z = float(lm.position[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.3
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            out.markers.append(m)
            n_published += 1

        self._cones_pub.publish(out)

        # Periodic diagnostic — every ~5 s of /Conos publishes,
        # surface the filter stats so the operator can confirm the
        # gate is doing what it should (filtering some, keeping most).
        if not hasattr(self, "_conos_log_counter"):
            self._conos_log_counter = 0
        self._conos_log_counter += 1
        if self._conos_log_counter % 50 == 0:
            self.get_logger().info(
                f"CONOS_FILTER published={n_published} "
                f"filtered={n_filtered} (min_obs={min_obs})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ConeGraphSlamNode()
    # NOTE: We tried MultiThreadedExecutor + callback groups to drain
    # IMU samples while the cone callback ran iSAM2 (the "skip scan: no
    # IMU samples" warnings suggested the executor was bottlenecked).
    # That broke standstill drastically (77 m drift in 28 s of being
    # parked at origin) — GTSAM's iSAM2 holds non-thread-safe state
    # even when our two callback groups never run graph code in
    # parallel. Stick with single-threaded until we have a way to
    # offload IMU buffering without touching the graph from a second
    # thread (e.g. a separate node + IPC).
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
