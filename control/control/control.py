from math import cos, sin
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from fs_msgs.msg import ControlCommand
from geometry_msgs.msg import TransformStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Empty
from visualization_msgs.msg import MarkerArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from transforms3d.euler import quat2euler

from control.controlador_stanley import StanleyController
from control.utils import wrap_to_pi
from control.velocity_control import VelocityControl


# TODO: general todos for this file are implementing config files to avoid hardcoded values
class Control(Node):
    """ROS2 node publishing control commands for the race vehicle."""

    PUBLISH_RATE_HZ: float = 40.0
    QUEUE_SIZE: int = 10

    def __init__(self) -> None:
        super().__init__("Control")
        self._init_params()
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_tf()
        self._init_variables()
        self._init_controllers()
        self._setup_timer()

    def _setup_publishers(self) -> None:
        """Create the publishers responsible for emitting vehicle commands."""
        self.command_publisher = self.create_publisher(
            ControlCommand, "/control_command", self.QUEUE_SIZE
        )
        # One-shot, latched: bridge picks this up to engage the real-car EBS
        # analog (setCarControls 0 0 1 + disableApiControl in UE5). Once
        # fired the bridge drops any further ControlCommand so the brake
        # cannot be released by a residual autonomy message.
        ebs_qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )
        self.ebs_publisher = self.create_publisher(Empty, "/signal/ebs", ebs_qos)
        # Companion topic: tell the bridge to clear its latched EBS flag so a
        # fresh session always starts with controls accepted. Without this, an
        # autonomous-stop at the end of session N (line ~424 below) would
        # leave the bridge dropping every setCarControls in session N+1 until
        # the user manually restarted dv_pipeline_stack. Published once below; the
        # bridge listens with the same TRANSIENT_LOCAL QoS so a
        # subscription-after-publish still picks it up.
        self.ebs_reset_publisher = self.create_publisher(
            Empty, "/signal/ebs_reset", ebs_qos
        )
        self.ebs_reset_publisher.publish(Empty())

    def _setup_subscribers(self) -> None:
        """Subscribe to the topics providing path, velocity, and odometry feedback."""
        self.path_subscriber = self.create_subscription(
            Path, "Path", self.path_callback, self.QUEUE_SIZE
        )
        # /gss and /testing_only/odom are published BEST_EFFORT by the
        # bridge (high-rate sensor stream — drops are correct, backpressure
        # is not). RELIABLE subscribers will silently fail to connect to
        # BEST_EFFORT publishers in ROS 2, so match the QoS here.
        sensor_qos = QoSProfile(
            depth=self.QUEUE_SIZE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.velocity_subscriber = self.create_subscription(
            TwistWithCovarianceStamped,
            "/fsds/gss",
            self.velocity_callback,
            sensor_qos,
        )
        self.odom_subscriber = self.create_subscription(
            Odometry,
            "/fsds/testing_only/odom",
            self.odometry_callback,
            sensor_qos,
        )
        # Big-orange cones detected by Cone_Detection on measured cluster
        # height (505 mm big-orange vs 325 mm small — see DS Table 1). Cones
        # are published in the fsds/FSCar frame, i.e. already vehicle-
        # relative, so forward distance is just position.x.
        self.orange_subscriber = self.create_subscription(
            MarkerArray,
            "/Conos_Orange",
            self.orange_callback,
            self.QUEUE_SIZE,
        )

    def _setup_tf(self) -> None:
        """Construct the TF buffer and listener used to query frame transforms."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _init_variables(self) -> None:
        """Initialise the state variables consumed by the control loop."""
        self.path: Path = Path()
        self.velocity: float = 0.0
        self.lateral_velocity: float = 0.0
        self.steering: float = 0.0
        # Big-orange cones in the current LiDAR frame (fsds/FSCar), i.e. the
        # position is already relative to the vehicle: x is forward distance.
        # Replaced wholesale on every MarkerArray message from Cone_Detection
        # (including empty messages), so this never goes stale.
        self.orange_cones: List[Tuple[float, float]] = []
        # Distance past the closest forward big-orange cone where the car
        # should target the end of its braking.
        #
        # History:
        #   - Hydraulic-brake era: 3 m was landing ~84.8 m (1 m past y=82
        #     finish cones).
        #   - fix/21 with the broken drag-only "EBS": raised to 5 m so
        #     the car would reach the Stop Area before coasting to a stop.
        #   - fix/22 with real regen + all-four EBS: the car now has
        #     plenty of brake authority, so 5 m pushed the landing back
        #     to ~89 m. 3 m lands the car inside the 1 m Stop Area
        #     again.
        self.stop_margin_m: float = 3.0
        # Start-gate-passed heuristic: treat orange ahead as a finish / lap
        # marker only after the car has driven at least this far from its
        # initial odom pose. Acceleration's start gate is ~7 m long, so 10 m
        # of travel reliably clears it. The latch-on-behind approach failed
        # because the LiDAR (mounted 1.4 m forward on the car) loses start-
        # gate cones out of its ±60° H-FOV before they ever reach x < 0.
        self.start_gate_travel_m: float = 10.0
        self.initial_position: Optional[Tuple[float, float]] = None
        self.distance_traveled: float = 0.0
        # Latched stop target in the traveled-distance reference frame. Once
        # we've sighted a big-orange finish gate the target locks in: the
        # car must stop when self.distance_traveled reaches this value.
        # Prevents the late-stage FoV dropout from unlatching the cap and
        # letting the car re-accelerate into the barrier past the gate.
        # Monotonic minimum — refinements can only pull the target closer.
        self.latched_stop_traveled: Optional[float] = None
        self.ebs_requested: bool = False

        # Continuity state across control ticks. The path planner publishes a
        # fresh /Path every tick (path_planning.py: ~30-pt resample of fsd's
        # output). Each new path is geometrically close to the last on a
        # straight, but at curve entry the cones in view shift and the new
        # path's nearest-point-to-vehicle can land several metres further
        # along than the old one. Stanley's argmin then jumps forward, the
        # heading at the new index is very different, yaw_err steps, and
        # steering saturates. We anchor the target index across solves by
        # tracking the previous tick's target world location and clipping
        # the new index to forward-only motion within physical limits.
        self._prev_target_world: Optional[Tuple[float, float]] = None
        # Steering rate-limit backstop. Real steering actuators slew at a
        # finite rate; software limit also smooths transients when the
        # anchor above can't fully suppress a discontinuity.
        self._prev_steering_cmd: float = 0.0
        # Planner-warm gate: hold the car stationary until the planner has
        # produced the same-shape path on at least two consecutive ticks.
        # The pre-go DIAG capture showed yaw_err = -89.7° / xte = +4.59 m
        # before the first real solve — benign only because v_tgt was 0.
        self._prev_path_signature = None
        self._path_stable_count: int = 0

    def _setup_timer(self) -> None:
        """Schedule the periodic execution of the control loop callback."""
        self.timer = self.create_timer(
            1.0 / self.PUBLISH_RATE_HZ, self.control_loop_callback
        )

    def _init_params(self) -> None:
        """Declare the ROS parameters that configure the controller gains."""

        # Stanley controller parameters.
        #   control_gain: was 4.0; reduced because the old wheelbase default of
        #     3.0 m was ~2× the real FS car value and was artificially inflating
        #     the projected cross-track error, effectively amplifying this gain.
        #   steering_damp_gain: was 1.0 — but the formula in
        #     StanleyController.stanley_control() is:
        #         output = desired - k_damp * (desired - prev)
        #               = (1 - k_damp) * desired + k_damp * prev
        #     so k_damp = 1.0 makes output = prev, which starts at 0 and is
        #     fed back as the next "prev", locking steering at 0 forever.
        #     Set to 0.0 (no damping) until we see the oscillation return.
        #     Acceleration didn't expose this because a straight path gives
        #     `desired ≈ 0` so output ≈ 0 was coincidentally correct.
        #     Autocross with curved paths surfaced it immediately — car drove
        #     dead-straight through every curve until off-track.
        # control_gain lowered from 2.5 → 1.0 after first autocross test:
        # at v=3 m/s with cross-track error of 2 m, the cross-track term
        #   atan2(k * e, k_soft + v) = atan2(5, 9) ≈ 29°
        # saturated max_steer (25°) → full lock → car turned too aggressively
        # one direction. With k=1.0 the same case gives ~12°, much gentler.
        # Tune up only if the car under-corrects on cleaner tracks.
        #
        # Bumped 1.0 → 1.5 on fix/49 after the clean-baseline drive: trajectory
        # recorder showed Stanley navigating the first curve correctly but
        # drifting outward to the yellow line at the apex (consistent with
        # underpowered cross-track at v ≈ 7 m/s, where atan2(k·e, k_soft+v)
        # damps to ~half its low-speed value). 1.5 lifts the high-speed
        # cornering term ~50% without re-triggering the saturation problem
        # that motivated the original 1.0 — at v=3, e=2 it still gives ~17°,
        # not the 29° saturation that broke it before.
        #
        # Bumped 1.5 → 2.0 after the max_normal_accel=5 drive still showed
        # the car kissing the outer (blue) corridor edge at the curve apex.
        # Chassis still has grip headroom (circle test: peak 6.4 m/s², we
        # use 5), so the right move is to ask Stanley for more steer at
        # any given xte rather than trade lap time for margin. At v=3,
        # e=2 the term is now atan2(4, 9) ≈ 24° — close to but not at
        # the 25° saturation, leaving the controller useful authority
        # in the worst case while doubling the high-speed cornering
        # response (atan2(1, 10) at v=7, e=0.5 ≈ 5.7° vs the 4.3° before).
        #
        # Reverted 2.0 → 1.5 on 2026-04-26 after the k=2.0 / k_soft=3 /
        # max_normal_accel=5 drive: car oscillated -0.90 → +0.29 → +0.58
        # in 4s and stalled at (17, -18) on the first curve. Combination of
        # higher gain + lower k_soft made low-speed startup hyperactive
        # (at v=1, atan2(2·e, 4) saturates on tiny errors). 1.5 keeps the
        # high-speed cornering lift from k_soft=3 without the LF instability.
        self.declare_parameter("control_gain", 1.5)
        # softening_gain dropped 6.0 → 3.0 on fix/49 after the control_gain=1.5
        # drive still showed outward drift at the top of the first curve.
        # Stanley's cross-track formula `atan2(k·e, k_soft + v)` damps with v;
        # at v=7 m/s, k=1.5, e=0.5 m the term was only ~3.3°, dominated by
        # k_soft=6. Halving k_soft doubles the high-speed cornering authority
        # without touching low-speed behaviour (where v ≪ k_soft so the term
        # is already saturated by the path heading anyway).
        self.declare_parameter("softening_gain", 3.0)
        self.declare_parameter("yaw_rate_gain", 0.0)
        self.declare_parameter("steering_damp_gain", 0.0)
        # Wheelbase for the Stanley front-axle projection. Authoritative
        # value from the IFS-08 Susp_Geometry sheet (CAD): 1600 mm. Matches
        # MODEL_IFS_08/SIMSCAPE and the chassis hardpoint data; the
        # previous 1550 mm was a vintage estimate, and the Chaos default
        # of 3000 mm was outright wrong.
        self.declare_parameter("wheelbase", 1.60)

        # Max steering angle
        self.declare_parameter("max_steering_ang", 25.0)

        # Velocity control parameters
        self.declare_parameter("Kp_vel", 0.3)
        self.declare_parameter("Kd_vel", 0.0)
        self.declare_parameter("Ki_vel", 0.01)

        # Feedforward gain for velocity control
        # Refer to velocity control for explanation
        self.declare_parameter("Fg", 1.0)

        self.control_gain = self.get_parameter("control_gain").value
        self.softening_gain = self.get_parameter("softening_gain").value
        self.yaw_rate_gain = self.get_parameter("yaw_rate_gain").value
        self.steering_damp_gain = self.get_parameter("steering_damp_gain").value
        self.wheelbase = self.get_parameter("wheelbase").value

        self.max_steering_ang = self.get_parameter("max_steering_ang").value

        self.Kp_vel = self.get_parameter("Kp_vel").value
        self.Kd_vel = self.get_parameter("Kd_vel").value
        self.Ki_vel = self.get_parameter("Ki_vel").value

        self.Fg = self.get_parameter("Fg").value

    def _init_controllers(self) -> None:
        """Instantiate helper controllers for steering and throttle/brake actions."""
        self.stanley_controller = StanleyController(
            control_gain=self.control_gain,
            softening_gain=self.softening_gain,
            yaw_rate_gain=self.yaw_rate_gain,
            steering_damp_gain=self.steering_damp_gain,
            max_steer=np.deg2rad(self.max_steering_ang),
            wheelbase=self.wheelbase,
        )
        self.velocity_controller = VelocityControl(
            self.Kp_vel, self.Ki_vel, self.Kd_vel, self.Fg
        )

    def velocity_callback(self, msg: TwistWithCovarianceStamped) -> None:
        """Record the latest longitudinal velocity measurement from the estimator."""
        self.velocity = msg.twist.twist.linear.x

    def path_callback(self, msg: Path) -> None:
        """Persist the most recent reference path for downstream control logic."""
        self.path = msg

    def odometry_callback(self, msg: Odometry) -> None:
        """Record the lateral component of velocity supplied by odometry."""
        self.lateral_velocity = msg.twist.twist.linear.y

    def orange_callback(self, msg: MarkerArray) -> None:
        """Cache the current frame's big-orange cones (car-relative coords)."""
        # Skip the sentinel DELETEALL marker Cone_Detection prepends
        # (marker.action == 3) — it carries no real position.
        self.orange_cones = [
            (float(m.pose.position.x), float(m.pose.position.y))
            for m in msg.markers
            if m.action != 3
        ]

    def _compute_orange_stop_distance(self) -> Optional[float]:
        """Distance from vehicle to the Stop Area target derived from the
        big-orange finish-gate markers.

        Only active after the car has travelled far enough from its
        starting pose to have cleared the start gate — before that, any
        orange ahead is assumed to be the start gate and we don't brake.

        Once a finish gate is sighted, we LATCH the stop target in the
        car's traveled-distance frame. Subsequent detections can only pull
        it closer (take the min). That stays sticky even when the oranges
        leave the LiDAR FoV in the last few metres before the gate — the
        previous implementation released the cap at that point and the
        velocity controller re-accelerated into the barrier."""
        if self.distance_traveled < self.start_gate_travel_m:
            return None

        # Refine the latch from the current orange detection (closest forward
        # cone). Require at least 5 m of forward distance to weed out LiDAR
        # returns off the car's own structure (wheels / aero at close range
        # measure tall enough to trip the big-orange height threshold).
        ORANGE_MIN_FORWARD_M = 5.0
        # The INITIAL latch must land at least this far along the travelled-
        # distance axis. Prevents the residual start-gate oranges from
        # latching the stop way before the finish (they were still being
        # reported near the car after the start-gate-travel check cleared).
        # Finish gates on any reasonable event are comfortably beyond 30 m
        # from spawn; refinements after latching can pull the target in.
        MIN_INITIAL_LATCH_TRAVELED_M = 30.0
        if self.orange_cones:
            forwards_ahead = [ox for ox, _ in self.orange_cones if ox > ORANGE_MIN_FORWARD_M]
            if forwards_ahead:
                candidate = self.distance_traveled + min(forwards_ahead) + self.stop_margin_m
                if self.latched_stop_traveled is None:
                    if candidate >= MIN_INITIAL_LATCH_TRAVELED_M:
                        self.latched_stop_traveled = candidate
                elif candidate < self.latched_stop_traveled:
                    self.latched_stop_traveled = candidate

        if self.latched_stop_traveled is None:
            return None
        remaining = self.latched_stop_traveled - self.distance_traveled
        return max(0.0, remaining)

    def control_loop_callback(self) -> None:
        """Run the primary control algorithm responsible for steering and speed."""
        command_msg = ControlCommand()

        # Pull the latest vehicle pose transform; a missing transform aborts this tick
        odom_to_vehicle, vehicle_to_odom = self.get_transforms()

        # If transforms are not available, skip this iteration
        if odom_to_vehicle is None or vehicle_to_odom is None:
            self.get_logger().warn(
                "Transforms not available, skipping control iteration"
            )
            return

        # Convert the vehicle orientation from quaternion to yaw (heading)
        yaw = quat2euler(
            [
                odom_to_vehicle.transform.rotation.w,
                odom_to_vehicle.transform.rotation.x,
                odom_to_vehicle.transform.rotation.y,
                odom_to_vehicle.transform.rotation.z,
            ]
        )[2]

        # Derive the front axle pose to better align control with the steering axle
        cg_pose = [
            odom_to_vehicle.transform.translation.x,
            odom_to_vehicle.transform.translation.y,
            yaw,
        ]
        front_axle_pose = self.get_front_axle_pose(cg_pose)

        # Collect path points that lie ahead of the vehicle to use as local reference
        forward_path_points = self.get_points_ahead(front_axle_pose)
        if not forward_path_points:
            command_msg.brake = 1.0
            self.command_publisher.publish(command_msg)
            return

        # Extract arrays of path positions and headings for the Stanley controller
        px = np.array([p.pose.position.x for p in self.path.poses])
        py = np.array([p.pose.position.y for p in self.path.poses])
        pyaw = np.array(
            [
                quat2euler(
                    [
                        p.pose.orientation.w,
                        p.pose.orientation.x,
                        p.pose.orientation.y,
                        p.pose.orientation.z,
                    ]
                )[2]
                for p in self.path.poses
            ]
        )

        # Planner-warm gate. The first one-or-two solves of every session
        # produce a path that doesn't reflect the actual cones in view, so
        # hold neutral until the path geometry is stable across at least
        # two consecutive ticks. Cheap signature: number of poses + the
        # rounded first-pose location.
        path_signature = (
            len(self.path.poses),
            round(float(px[0]), 1),
            round(float(py[0]), 1),
        )
        if path_signature == self._prev_path_signature:
            self._path_stable_count += 1
        else:
            self._path_stable_count = 0
        self._prev_path_signature = path_signature
        if self._path_stable_count < 2 and self.velocity < 0.5:
            command_msg.throttle = 0.0
            command_msg.brake = 0.05
            command_msg.steering = 0.0
            self.command_publisher.publish(command_msg)
            return

        # Feed the local path into the Stanley controller and obtain the steering demand
        self.stanley_controller.set_path(px, py, pyaw)

        # Arc-length anchor across path solves. We re-project the previous
        # target's world location onto the new path each tick and constrain
        # the new target index to the forward window
        # `[idx_of_prev_on_new_path, idx_of_prev_on_new_path + max_step]`
        # where `max_step` is the number of indices the vehicle can have
        # physically advanced this tick (`v · dt` plus a 1 m safety buffer).
        # The natural argmin is then clipped into this window. On a straight
        # this is a no-op (argmin == prev + 1); on the curve replans that
        # tripped the previous run, the leap is suppressed and yaw_err / xte
        # change continuously.
        nat_target, _ndx, _ndy, _nabs = self.stanley_controller.find_target_path_id(
            float(odom_to_vehicle.transform.translation.x),
            float(odom_to_vehicle.transform.translation.y),
            float(yaw),
        )
        nat_target = int(nat_target)
        if self._prev_target_world is None or len(px) <= 1:
            target_index = nat_target
        else:
            d_prev = np.hypot(
                px - self._prev_target_world[0], py - self._prev_target_world[1]
            )
            idx_prev = int(np.argmin(d_prev))
            ds = np.hypot(np.diff(px), np.diff(py))
            avg_ds = float(np.mean(ds)) if ds.size and float(np.mean(ds)) > 1e-3 else 0.7
            max_step_m = max(0.1, abs(self.velocity)) / self.PUBLISH_RATE_HZ + 1.0
            max_step_idx = max(1, int(np.ceil(max_step_m / avg_ds)))
            target_index = int(
                np.clip(nat_target, idx_prev, min(idx_prev + max_step_idx, len(px) - 1))
            )

        (limited_steering_angle, _target_index, crosstrack_error) = (
            self.stanley_controller.stanley_control_at_index(
                float(odom_to_vehicle.transform.translation.x),
                float(odom_to_vehicle.transform.translation.y),
                float(yaw),
                self.velocity,
                target_index,
                self.steering,
            )
        )
        self._prev_target_world = (
            float(px[target_index]),
            float(py[target_index]),
        )
        # Update the travelled-distance counter used by the orange stop gate.
        cx = float(odom_to_vehicle.transform.translation.x)
        cy = float(odom_to_vehicle.transform.translation.y)
        if self.initial_position is None:
            self.initial_position = (cx, cy)
        self.distance_traveled = float(
            np.hypot(cx - self.initial_position[0], cy - self.initial_position[1])
        )

        # Calibration-period diagnostic: log orange-stop state once a second so
        # we can reconstruct what the autonomous-stop gate saw during a run.
        # Remove once the kinematic cap is validated end-to-end.
        now_ns = self.get_clock().now().nanoseconds
        if not hasattr(self, "_last_log_ns") or (now_ns - self._last_log_ns) > 1_000_000_000:
            self._last_log_ns = now_ns
            fwds = [ox for ox, _ in self.orange_cones if ox > 0.0]
            min_fwd = f"{min(fwds):.1f}m" if fwds else "n/a"
            max_fwd = f"{max(fwds):.1f}m" if fwds else "n/a"
            latch = f"{self.latched_stop_traveled:.1f}m" if self.latched_stop_traveled is not None else "none"
            self.get_logger().info(
                f"AUTOSTOP: traveled={self.distance_traveled:.1f}m "
                f"orange_n={len(self.orange_cones)} min_fwd={min_fwd} max_fwd={max_fwd} "
                f"latch={latch} v={self.velocity:.1f}m/s"
            )

        # Distance to the stop target for the velocity controller's kinematic
        # cap. Order of preference:
        #   1. Big-orange finish gate (farthest orange ahead + 3 m margin).
        #      This is the FS Driverless Spec definition of the Stop Area.
        #      Only active once the car has driven past the start gate.
        #   2. Fallback: last pose of the perceived path — handles SLAM
        #      dropouts and events with no orange ahead.
        # `is_stop_area` flag tells the velocity controller whether the
        # distance refers to a real orange Stop Area (so it can engage its
        # monotonic-decrease cap) or just the end of a short perceived
        # path (where the cap would lock v_tgt at 0 and never release).
        orange_stop_distance = self._compute_orange_stop_distance()
        is_stop_area = orange_stop_distance is not None
        distance_to_path_end = orange_stop_distance
        if distance_to_path_end is None and self.path.poses:
            last_pose = self.path.poses[-1].pose.position
            dx = last_pose.x - float(odom_to_vehicle.transform.translation.x)
            dy = last_pose.y - float(odom_to_vehicle.transform.translation.y)
            distance_to_path_end = float(np.hypot(dx, dy))

        # Compute the required longitudinal command given the target profile
        velocity_command_value, velocity_ref = (
            self.velocity_controller.get_control_value(
                self.velocity, forward_path_points, crosstrack_error,
                distance_to_path_end=distance_to_path_end,
                is_stop_area=is_stop_area,
            )
        )

        # Acceleration and braking are limited inside the velocity controller.
        # Explicit zero on the opposite channel so we never send throttle and
        # brake together (FS rule D 10.1.7 / safety).
        if velocity_command_value > 0:
            command_msg.throttle = velocity_command_value
            command_msg.brake = 0.0
        else:
            command_msg.throttle = 0.0
            command_msg.brake = -velocity_command_value

        # EBS trigger: safety net. Primary stopping is done by the velocity
        # controller's regen cap; EBS fires only when the car is close
        # enough to the latched stop AND fast enough that we can't finish
        # braking on regen alone. Once fired the bridge calls `activateEbs`
        # which clamps the handbrake on all four wheels and locks out
        # future control inputs until reset.
        #
        # The 25 m remaining-distance gate is deliberate: the latch starts
        # coarse (one far-sighted orange cone → ~89 m) and only refines
        # inward to the true finish (~85 m) once more cones come into view
        # around traveled=70 m. Triggering EBS before that refinement
        # locks in a stale, too-far stop target and the car halts short.
        #
        # ebs_decel = 11 m/s² is calibrated to fix/22's all-four-wheel
        # EBS bench (bench_ebs_rpc.py: v=17 m/s → 0 in 12.3 m, avg
        # -11.03 m/s². Previously the "EBS" path was effectively a drag
        # coast (~2 m/s²) because of the keyboard-override bug that
        # fix/22 eliminated — the old 1.9 m/s² constant here was
        # compensating for that broken path, not for real EBS behavior.
        # Pneumatic full-lock is tire-grip-limited, so 11 m/s² is the
        # physical ceiling; the 0.5 m safety margin absorbs tick latency.
        ebs_decel = 11.0
        ebs_remaining = (
            self.latched_stop_traveled - self.distance_traveled
            if self.latched_stop_traveled is not None
            else float("inf")
        )
        ebs_stop_dist = (self.velocity * self.velocity) / (2.0 * ebs_decel)
        if (
            self.latched_stop_traveled is not None
            and ebs_remaining < 25.0
            and self.distance_traveled
            >= self.latched_stop_traveled - ebs_stop_dist - 0.5
        ):
            command_msg.throttle = 0.0
            command_msg.brake = 1.0
            if not self.ebs_requested:
                self.ebs_publisher.publish(Empty())
                self.ebs_requested = True
                self.get_logger().info("EBS requested — autonomous stop engaged")

        # Gate steering: don't apply Stanley steering until the car is moving.
        # At near-zero speed the cross-track term saturates to max lock, causing
        # the car to spin before it has any forward momentum. (Tried lifting
        # this gate when xte > 0.5 to enable stuck-recovery; backfired at
        # curve entry — when v_tgt drops with cross-track penalty and v
        # crosses below 0.5 mid-corner, full-lock fires and the car spins
        # through the cones. Stuck recovery needs a different approach.)
        if abs(self.velocity) < 0.5:
            self.steering = 0.0
        else:
            # Sign convention reasoning (corrected after live drive in fix/46
            # validation):
            #
            #   The Stanley implementation in StanleyController follows the
            #   Stanford-paper convention where the OUTPUT semantics are
            #   "positive = LEFT turn" (CCW). The cross-track formula's
            #   `front_axle_vector = (sin yaw, -cos yaw)` points to the car's
            #   right, so a path on the LEFT of the car gives a positive
            #   `crosstrack_error` — and the formula then returns a positive
            #   `crosstrack_steering_error` meaning "turn LEFT toward the
            #   path." Same with `yaw_error`: positive when path heads more
            #   CCW than vehicle, meaning "turn LEFT." Both terms agree.
            #
            #   UE5's `setSteeringInput` uses the opposite convention:
            #   positive = RIGHT. So we MUST negate Stanley's output to
            #   match UE5.
            #
            #   fix/43 mistakenly dropped this negation thinking the
            #   front_axle_vector pointing right meant the OUTPUT was
            #   "positive=right". It isn't — that vector is just the
            #   reference frame for measuring crosstrack, the output
            #   convention is still standard-Stanley. Re-instating the
            #   negation here.
            self.steering = -limited_steering_angle

        # Steering slew-rate limit. Real-car servo + tie-rod system slews at
        # roughly 500–700 °/s; we clamp tighter (360 °/s ⇒ 0.157 rad/tick at
        # 40 Hz) to absorb residual transients the arc-length anchor above
        # cannot suppress (e.g. a single mis-detected cone shifting the
        # planned path's curvature for one tick). A clean lap should never
        # hit this limit — it's only a backstop.
        max_steer_rate_rad_per_tick = np.deg2rad(360.0) / self.PUBLISH_RATE_HZ
        self.steering = float(np.clip(
            self.steering,
            self._prev_steering_cmd - max_steer_rate_rad_per_tick,
            self._prev_steering_cmd + max_steer_rate_rad_per_tick,
        ))
        self._prev_steering_cmd = self.steering

        command_msg.steering = self.steering / np.deg2rad(self.max_steering_ang)

        self.command_publisher.publish(command_msg)

        # === DIAGNOSTIC LOGGING (every 8 ticks ≈ 5 Hz at 40 Hz publish) ===
        # Captures the key signals to understand what the controller "sees"
        # at the moment it commands a steering/throttle decision. Used to
        # diagnose high-speed curve failures: is the path shrinking on
        # turn-in? does the kinematic stop cap kick in early? does the
        # cross-track jump suddenly when the path re-plans?
        self._diag_tick = getattr(self, '_diag_tick', 0) + 1
        if self._diag_tick % 8 == 0:
            # Cumulative distance along forward_path_points — actual
            # forward path the controller is following right now.
            path_len_m = 0.0
            for i in range(1, len(forward_path_points)):
                dx_p = forward_path_points[i][0] - forward_path_points[i-1][0]
                dy_p = forward_path_points[i][1] - forward_path_points[i-1][1]
                path_len_m += float(np.hypot(dx_p, dy_p))
            # Velocity controller exposes its filtered target speed.
            v_target = getattr(self.velocity_controller, 'filtered_target_speed', float('nan'))
            d_end_str = f"{distance_to_path_end:.1f}" if distance_to_path_end is not None else "n/a"
            self.get_logger().info(
                f"DIAG spd={self.velocity:5.2f} v_tgt={v_target:5.2f} "
                f"v_cmd={velocity_command_value:+5.2f} "
                f"thr={command_msg.throttle:.2f} brk={command_msg.brake:.2f} | "
                f"path_n={len(forward_path_points):2d} path_len={path_len_m:5.1f}m "
                f"d_end={d_end_str:>5}m | "
                f"xte={crosstrack_error:+5.2f}m yaw_err={float(np.degrees(wrap_to_pi(pyaw[_target_index] - yaw))):+6.1f}deg "
                f"stanley={float(np.degrees(limited_steering_angle)):+6.1f}deg "
                f"steer_cmd={command_msg.steering:+.3f}"
            )

    def get_transforms(
        self,
    ) -> Tuple[Optional[TransformStamped], Optional[TransformStamped]]:
        """Fetch the transforms linking the odom frame with the vehicle frame."""
        try:
            odom_to_vehicle = self.tf_buffer.lookup_transform(
                "odom", "fsds/FSCar", Time()
            )
        except TransformException as ex:
            # Convert exception to string for logging
            self.get_logger().warn(f"Transform lookup failed: {str(ex)}")
            return None, None

        try:
            vehicle_to_odom = self.tf_buffer.lookup_transform(
                "fsds/FSCar", "odom", Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"Inverse transform lookup failed: {str(ex)}")
            return None, None

        return odom_to_vehicle, vehicle_to_odom

    def get_points_ahead(self, vehicle_pose: Sequence[float]) -> List[List[float]]:
        """Return path points that sit in front of the vehicle, ordered by proximity."""

        # Get path points
        path_xy: List[List[float]] = [
            [p.pose.position.x, p.pose.position.y] for p in self.path.poses
        ]
        if not path_xy:
            return []

        # Build points array
        points_with_distances: List[Tuple[List[float], float]] = []
        vehicle_heading = vehicle_pose[2]

        for point in path_xy:
            dx = point[0] - vehicle_pose[0]
            dy = point[1] - vehicle_pose[1]
            distance = np.sqrt(dx * dx + dy * dy)
            angle_to_point = np.arctan2(dy, dx)

            if abs(wrap_to_pi(vehicle_heading - angle_to_point)) < np.pi * 3 / 4:
                points_with_distances.append((point, distance))

        # Sort points by distance and return only points
        return [p[0] for p in sorted(points_with_distances, key=lambda x: x[1])]

    def get_front_axle_pose(self, cg_pose: Sequence[float]) -> List[float]:
        """Approximate the front axle position using the centre-of-gravity pose."""

        x_axle = cg_pose[0] + cos(cg_pose[2]) * 0.5
        y_axle = cg_pose[1] + sin(cg_pose[2]) * 0.5

        return [x_axle, y_axle, cg_pose[2]]


"""ROS2 node helpers."""


def control(args: Optional[Sequence[str]] = None) -> None:
    """Entry point that initialises rclpy and spins the Control node."""
    rclpy.init(args=args)

    control_node = Control()
    rclpy.spin(control_node)
