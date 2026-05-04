"""Cone-graph SLAM node — PR B (cone observation factors + DA).

Subscribes:
    /imu               (sensor_msgs/Imu, BEST_EFFORT, ~400 Hz)
    /Conos_raw         (visualization_msgs/MarkerArray, base_link
                       frame, 10 Hz from Cone_Detection)

Publishes:
    /tf                (odom -> base_link)
    /cone_slam/state   (nav_msgs/Odometry — pose+velocity at base_link)
    /Conos             (visualization_msgs/MarkerArray, world-frame
                       cone map with persistent landmark IDs encoded
                       as marker.id)

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
from rclpy.node import Node
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
from std_msgs.msg import Float32
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
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


# Motor-RPM → body-frame longitudinal velocity.
#
# The doc-derived constant from docs/dv_pipeline_rebuild.md §3.5
# was 0.00821, computed as (2π × 0.228 / 2.909) / 60 (i.e.
# WheelRadius=0.228 m, GearRatio=2.909). On 2026-04-28 a full bag
# diagnostic against trackA_manual_001602's GT odometry showed the
# documented constant was 8.6 % too low — the actual mean ratio of
# rpm-derived speed to GT speed across 7000+ motion samples was
# 0.9144. The simulator's effective wheel circumference / gear-ratio
# product corresponds to a constant of 0.00821 / 0.9144 ≈ 0.00898
# (equivalently: an effective wheel radius of ~0.249 m if we trust
# the gear ratio, or an effective gear ratio of ~2.66 if we trust the
# wheel radius). Using the doc constant systematically pulled the
# velocity prior toward an underestimated speed → ~7 m of position
# underestimate over 80 m of driving. Empirical constant adopted.
RPM_TO_MS = 0.00898

# Drop /motor_rpm samples this old (seconds). Sustained RPM staleness
# means the bridge stopped publishing; fall back to no velocity prior
# rather than constraining the optimizer with last-good-but-stale data.
RPM_STALE_S = 0.5


class State(Enum):
    INIT_WAITING_IMU = 1
    INIT_CALIBRATING = 2
    SLAM_RUNNING = 3


class ConeGraphSlamNode(Node):
    def __init__(self) -> None:
        super().__init__("cone_graph_slam")

        # --- Frames (REP-105) ---
        self.odom_frame = self.declare_parameter(
            "odom_frame", "odom").value
        self.base_frame = self.declare_parameter(
            "base_frame", "base_link").value

        # --- Pose-jump sanity check thresholds (#273 follow-up) ---
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

        # --- Components ---
        self._preint = ImuPreintegrator()
        self._graph = FactorGraph()
        self._db = LandmarkDb()
        self._latest_result: Optional[ScanResult] = None
        self._state = State.INIT_WAITING_IMU
        self._calib_started_t: Optional[float] = None

        # Optional per-landmark creation diagnostic. When DV_SLAM_LANDMARK_CAPTURE
        # is set to a writable path, every new landmark dumps one JSON line
        # (id, body_x, body_y, range, color, pred_yaw_deg, step) at creation.
        # Used to verify the colour-lock-on-first-observation hypothesis (#188):
        # if many landmarks created at long range or during a yaw rotation
        # have body_y near the threshold and get locked the wrong colour, the
        # hypothesis is confirmed.
        import os as _os
        self._lm_capture_path = _os.environ.get("DV_SLAM_LANDMARK_CAPTURE", "")
        self._lm_capture_fh = None
        if self._lm_capture_path:
            try:
                self._lm_capture_fh = open(self._lm_capture_path, "w")
                self.get_logger().info(
                    f"DV_SLAM_LANDMARK_CAPTURE → {self._lm_capture_path}")
            except OSError as ex:
                self.get_logger().error(f"landmark capture open failed: {ex}")
                self._lm_capture_fh = None

        # Per-second SLAM observation diagnostic. Breaks every scan's
        # cones into (incoming, associated, new) per colour, plus
        # cascade-skipped scans. Used to find where yellow cones
        # disappear in tight corners (#189): if obs has 7 yellow but
        # assoc+new totals only 1 yellow, the loss is in SLAM (DA gate
        # / pose-jump rejection); if obs already has 1 yellow, the
        # loss is upstream in Cone_Detection.
        self._obs_diag = {}
        self._obs_n_scans = 0
        self._obs_last_log_ns = 0

        # Latest /motor_rpm sample. Wall-clock timestamp because the
        # bridge stamps it with node_->now() (no header.stamp on
        # std_msgs/Float32). Used for staleness check inside _on_cones.
        import time as _time
        self._time = _time
        self._latest_rpm: Optional[float] = None
        self._latest_rpm_t: Optional[float] = None

        # --- Subscriptions ---
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
        self.create_subscription(Imu, "/imu", self._on_imu, imu_qos)
        # /Conos_raw is the scan trigger AND the source of cone
        # observations. Cone_Detection publishes ~10 Hz, in base_link
        # frame, with cone height encoded on marker.scale.z.
        self.create_subscription(
            MarkerArray, "/Conos_raw",
            self._on_cones, 10)

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
        self.create_subscription(Float32, "/motor_rpm", self._on_rpm, rpm_qos)

        # --- /testing_only/odom — sim ground truth, used only for the
        # GT-aligned diagnostic published below. The bridge encodes this
        # in ENU (East-North-Up); SLAM internally anchors to the car's
        # initial pose, so direct comparison would mix two different
        # world frames. We snapshot the GT pose at calibration-end and
        # publish a pose-aligned residual on /cone_slam/gt_aligned.
        # Bridge publishes BEST_EFFORT (per `sensor_qos` in
        # ifssim_ros_wrapper.cpp:289); subscriber must match or messages
        # silently never arrive.
        gt_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Odometry, "/testing_only/odom", self._on_gt_odom, gt_qos)
        self._latest_gt: Optional[Odometry] = None
        self._gt_init_pose: Optional[gtsam.Pose3] = None

        # --- Publishers ---
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._state_pub = self.create_publisher(
            Odometry, "/cone_slam/state", 10)
        self._cones_pub = self.create_publisher(
            MarkerArray, "/Conos", 10)
        # GT-aligned diagnostic odometry. Same Odometry shape as
        # /cone_slam/state but containing the ground truth re-expressed
        # in SLAM's anchored body-frame world. SLAM-vs-GT divergence in
        # this frame is the actual SLAM drift; in the raw GT frame it
        # would be that drift plus the static frame mismatch.
        self._gt_aligned_pub = self.create_publisher(
            Odometry, "/cone_slam/gt_aligned", 10)
        # Scalar position-error magnitude — convenience for the
        # Lichtblick plot, equal to ||state.position - gt_aligned.position||.
        self._gt_error_pub = self.create_publisher(
            Float32, "/cone_slam/gt_error_m", 10)

        # Anchor `map -> odom` as identity on /tf_static. We don't
        # have map-based localization (no GPS-aligned global frame),
        # so the SLAM odom frame IS effectively the map frame for
        # downstream consumers. Without this static, Lichtblick's 3D
        # panel has no root and nothing renders — the cone map and
        # the trajectory both anchor on `odom`, which dangles off
        # /tf_static at recording time. The transform itself is
        # identity; the publisher exists purely to give visualizers
        # a parent frame to walk from.
        self._publish_map_to_odom_static()

        self.get_logger().info(
            "cone_graph_slam initialized — waiting for IMU "
            f"(odom='{self.odom_frame}', base='{self.base_frame}')")

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
        if (total_pre >= 5
                and self._graph.step > 30
                and n_new_pre > int(0.60 * total_pre)):
            self.get_logger().warn(
                f"skip cone factors: DA-failure spike "
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
            self._publish_tf(stamp, result)
            self._publish_state(stamp, result)
            self._publish_cone_map(stamp)
            per_scan["skipped"] = 1
            self._accumulate_obs_diag(per_scan)
            return

        # For each matched obs → factor between current pose and the
        # known landmark. For unmatched → allocate a new landmark and
        # add a factor to it.
        n_new = 0
        n_assoc = 0
        for o, m in zip(observations, matches):
            if m.landmark_id == -1:
                world_xyz = self._body_to_world(
                    o.body_x, o.body_y, pred_x, pred_y, pred_yaw)
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

        self._publish_tf(stamp, result)
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

    def _publish_tf(self, stamp, result: ScanResult) -> None:
        # Internal pose is Pose3 (6-DOF); we emit the TF as-is. The
        # decision in the design doc was to project to 2D for output,
        # but TransformStamped is naturally 3D — Foxglove / RViz still
        # render it correctly. We can flatten z/roll/pitch later if
        # downstream consumers care.
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        pose = result.pose
        t.transform.translation.x = pose.x()
        t.transform.translation.y = pose.y()
        t.transform.translation.z = pose.z()
        q = pose.rotation().toQuaternion()
        t.transform.rotation.w = q.w()
        t.transform.rotation.x = q.x()
        t.transform.rotation.y = q.y()
        t.transform.rotation.z = q.z()
        self._tf_broadcaster.sendTransform(t)

    def _publish_map_to_odom_static(self) -> None:
        """Emit `map -> odom` identity on /tf_static once at startup.

        Lichtblick (and rviz2) need a static root that the dynamic
        chain can hang off; without one the 3D panel renders nothing.
        Recording-time consumers see this single message at t=0 and
        keep it for the rest of the run, so it costs effectively
        nothing per scan.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = self.odom_frame
        t.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(t)

    def _publish_state(self, stamp, result: ScanResult) -> None:
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
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
        out.header.frame_id = self.odom_frame
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

        marker.id encodes the persistent SLAM landmark id so downstream
        consumers (path_planning) can track cone identity across scans.
        Color is mapped to RGB so RViz/Foxglove renders correctly
        without extra config.
        """
        if len(self._db) == 0:
            return

        out = MarkerArray()
        # First marker is DELETEALL so visualizers refresh cleanly.
        delete_all = Marker()
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = self.odom_frame
        delete_all.action = Marker.DELETEALL
        out.markers.append(delete_all)

        # Single neutral colour for every landmark — SLAM has no
        # per-cone colour anymore. Yellow for visibility on dark
        # backgrounds; the path planner ignores the colour anyway and
        # routes everything through ConeTypes.UNKNOWN (#268). The
        # per-cone marker.id still encodes the persistent landmark id
        # so downstream consumers (path_planning) can identify cones
        # across scans.
        for lm in self._db:
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.odom_frame
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

        self._cones_pub.publish(out)


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
