"""IFSSIM autonomy control node — clean rewrite (feat/34).

Single ROS node that:
  1. Subscribes to /Path (planner), /cone_slam/state (SLAM Odometry),
     /Conos_Orange (finish-gate detection)
  2. Builds VehicleState + ReferenceTrajectory each tick
  3. Calls a LateralController + LongitudinalController (selected by params)
  4. Publishes /control_command (fs_msgs/ControlCommand)
  5. Publishes /signal/ebs (latched) on operator/safety request

Keeps strategies behind ABCs so swapping Pure Pursuit → LQR or PI → MPC is one
line in the factory below. The node does no path geometry, no slip math, no
PID — those live in the strategy implementations.

Wire format note: the bridge translates ControlCommand{throttle, steering,
brake} → setVehicleCommand{throttle, steering, regen} (feat/33). We emit
explicit unsigned channels: ControlCommand.throttle is motor torque demand
[0,1] and ControlCommand.brake is regen demand [0,1]. The deadband inside
the longitudinal controller guarantees only one of them is non-zero.
"""
from __future__ import annotations
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time

from fs_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Empty, Float32
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from transforms3d.euler import quat2euler

from control.state import VehicleState
from control.reference import ReferenceTrajectory
from control.controllers.base import LateralController, LongitudinalController
from control.controllers.pure_pursuit import PurePursuit
from control.controllers.pi_velocity import PIVelocity


class ControlNode(Node):
    PUBLISH_RATE_HZ = 40.0

    def __init__(self) -> None:
        super().__init__("control")
        self._declare_params()
        self._build_strategies()

        # Latest input state, refreshed by callbacks
        self._latest_odom: Optional[Odometry] = None
        self._latest_path_xs: list[float] = []
        self._latest_path_ys: list[float] = []
        # Per-pose curvature from path_planning, smuggled through
        # `pose.position.z` (FaSTTUBe analytical κ; see path_planning's
        # _pose_stamped). Empty when no path has arrived yet; the
        # ReferenceTrajectory builder falls back to its own finite-
        # difference κ when this list is empty or all-zero.
        self._latest_path_kappas: list[float] = []
        # Big-orange forward distances (base_link frame) — captured at the
        # tick when the gate is first detected, then frozen as a stop anchor
        # in odom frame (see _on_orange / _stop_distance).
        self._stop_anchor_xy: Optional[tuple[float, float]] = None
        self._stop_latched: bool = False
        # Travel distance (odom-frame) used as the gate-latch guard. The FS
        # start gate is also big-orange, so a naive "latch on first sighting"
        # snaps onto the start cones and the controller immediately tries to
        # brake to a stop. The lap-min-distance guard suppresses the latch
        # until we've travelled past where the start cones could possibly be.
        self._origin_xy: Optional[tuple[float, float]] = None
        self._travelled: float = 0.0
        self._last_pose_xy: Optional[tuple[float, float]] = None

        # TF listener (cone_graph_slam publishes odom→base_link)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Publishers
        self._cmd_pub = self.create_publisher(ControlCommand, "/control_command", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._ebs_pub = self.create_publisher(Empty, "/signal/ebs", latched)
        # /signal/ebs_reset clears the bridge's `ebs_triggered_` gate, which
        # otherwise drops every /control_command silently after a previous
        # ActivateEbs (incl. mission_control's RES-activate at event_start
        # before the autonomy boots). Latched + retry: a single publish can
        # race the bridge's subscriber readiness, so we re-fire a few times
        # while the bridge is connecting.
        self._ebs_reset_pub = self.create_publisher(Empty, "/signal/ebs_reset", latched)
        self._ebs_reset_pub.publish(Empty())
        self._ebs_reset_retries = 4
        self._ebs_reset_timer = self.create_timer(0.5, self._republish_ebs_reset)

        # Diagnostic publishers (#260 follow-up). v_set tracks the
        # longitudinal controller's setpoint each tick; kappa_max
        # exposes the local curvature it's reacting to. Plotted
        # alongside SLAM v in slam_debug.json — if v_set doesn't drop
        # going into a corner, the velocity profile isn't being honoured
        # and that's why the car carries too much speed into the apex.
        self._v_set_pub = self.create_publisher(Float32, "/control/v_set_mps", 10)
        self._kappa_max_pub = self.create_publisher(
            Float32, "/control/kappa_max_per_m", 10)

        # Subscribers
        self.create_subscription(Path, "Path", self._on_path, 10)
        self.create_subscription(Odometry, "/cone_slam/state", self._on_odom, 10)
        self.create_subscription(MarkerArray, "/Conos_Orange", self._on_orange, 10)

        # Tick
        self.create_timer(1.0 / self.PUBLISH_RATE_HZ, self._tick)

        self.get_logger().info(
            f"control_node: lateral={self.lateral.__class__.__name__} "
            f"longitudinal={self.longitudinal.__class__.__name__} "
            f"@ {self.PUBLISH_RATE_HZ:.0f} Hz"
        )

    # ------------------------------------------------------------------ params

    def _declare_params(self) -> None:
        self.declare_parameter("lateral_controller", "pure_pursuit")
        self.declare_parameter("longitudinal_controller", "pi_velocity")
        # Tunables — passed to the strategy constructors. Keep flat: each
        # strategy reads what it needs, ignores the rest.
        self.declare_parameter("v_max", 3.0)  # tuning experiment — completable-lap baseline
        # a_lat_max governs corner braking: v_corner = sqrt(R · a_lat_max).
        # 6.0 m/s² (~0.6 g lateral) was the original default — fine for
        # a real FS car on slicks, too generous for the IFS-08 sim's
        # tire envelope. With a_lat_max=6 and v_max=3, even a 1.5 m
        # radius hairpin yields v_corner ≥ v_max → setpoint never drops
        # → controller carries v_max into the apex and goes off (#260
        # follow-up). Dropped to 3.0 m/s² (~0.3 g): v_corner < v_max
        # whenever R < 3 m, so any FS-style hairpin actually triggers
        # a setpoint drop. Tune up later once tire physics is known.
        self.declare_parameter("a_lat_max", 3.0)
        self.declare_parameter("a_dec_max", 4.0)
        self.declare_parameter("lookahead_min", 1.5)
        self.declare_parameter("lookahead_k", 0.5)
        self.declare_parameter("kp_v", 0.5)
        self.declare_parameter("ki_v", 0.05)
        self.declare_parameter("deadband_v", 0.2)
        self.declare_parameter("throttle_max", 0.6)
        # Stop-latch guard. The FS start gate is also big-orange, so we
        # need to drive at least one lap-ish before the first orange
        # detection counts as the finish. Trackdrive courses are >100 m
        # per lap; 30 m is safely past any start-gate proximity.
        self.declare_parameter("stop_latch_min_travel", 30.0)

    def _p(self, name: str):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ factory

    def _build_strategies(self) -> None:
        """Pick controller implementations from params. Adding LQR/MPC later
        is one new file under controllers/ + one branch in this factory."""
        lat_name = self._p("lateral_controller")
        lon_name = self._p("longitudinal_controller")

        self.lateral: LateralController
        if lat_name == "pure_pursuit":
            self.lateral = PurePursuit(
                lookahead_min=self._p("lookahead_min"),
                lookahead_k=self._p("lookahead_k"),
            )
        else:
            raise ValueError(f"unknown lateral_controller={lat_name!r}")

        self.longitudinal: LongitudinalController
        if lon_name == "pi_velocity":
            self.longitudinal = PIVelocity(
                v_max=self._p("v_max"),
                a_lat_max=self._p("a_lat_max"),
                a_dec_max=self._p("a_dec_max"),
                kp=self._p("kp_v"),
                ki=self._p("ki_v"),
                deadband=self._p("deadband_v"),
                throttle_max=self._p("throttle_max"),
            )
        else:
            raise ValueError(f"unknown longitudinal_controller={lon_name!r}")

    # ----------------------------------------------------------- subscriptions

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _on_path(self, msg: Path) -> None:
        self._latest_path_xs = [p.pose.position.x for p in msg.poses]
        self._latest_path_ys = [p.pose.position.y for p in msg.poses]
        # Side-channel: pose.position.z carries the planner's per-pose
        # curvature (FaSTTUBe analytical κ). See path_planning's
        # _pose_stamped for the encoding rationale.
        self._latest_path_kappas = [p.pose.position.z for p in msg.poses]

    def _on_orange(self, msg: MarkerArray) -> None:
        """Latch a stop anchor on the first tick we see ≥2 big-orange cones,
        AFTER the car has travelled at least stop_latch_min_travel metres
        from origin. The minimum-travel gate exists because the FS start
        gate is also big-orange; without it, we latch on the start cones
        at t=0 and the controller immediately tries to brake to a stop.

        Once latched, never unlatch — transient detector flicker (audit
        P0 item #2) cannot release the stop. The anchor is the centroid
        of the orange cones, projected from the car's current odom-frame
        pose. After that, _stop_distance() is the residual euclidean
        distance from the car to the anchor in odom frame.

        The cones come in base_link (vehicle frame), so we transform via
        the latest odom pose — close enough at the moment of latch since
        the car is still ~10 m from the gate."""
        if self._stop_latched or self._latest_odom is None:
            return
        if len(msg.markers) < 2:
            return
        if self._travelled < self.get_parameter("stop_latch_min_travel").value:
            return
        # Centroid in base_link
        n = len(msg.markers)
        sx = sum(m.pose.position.x for m in msg.markers) / n
        sy = sum(m.pose.position.y for m in msg.markers) / n
        # base_link → odom using current odom pose
        o = self._latest_odom
        q = o.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        ax = o.pose.pose.position.x + cos_y * sx - sin_y * sy
        ay = o.pose.pose.position.y + sin_y * sx + cos_y * sy
        self._stop_anchor_xy = (ax, ay)
        self._stop_latched = True
        self.get_logger().info(
            f"stop latched at odom=({ax:.2f}, {ay:.2f}) "
            f"from {n} big-orange cones"
        )

    def _republish_ebs_reset(self) -> None:
        """Re-fire /signal/ebs_reset for a short window after init to win the
        race against the bridge's subscriber matching. Self-cancels."""
        if self._ebs_reset_retries <= 0:
            self._ebs_reset_timer.cancel()
            return
        self._ebs_reset_pub.publish(Empty())
        self._ebs_reset_retries -= 1

    def _stop_distance(self, state: VehicleState) -> float:
        """Euclidean distance from car to the latched stop anchor. Returns
        +inf when no stop is latched (controller treats this as 'no cap')."""
        if not self._stop_latched or self._stop_anchor_xy is None:
            return float("inf")
        ax, ay = self._stop_anchor_xy
        return math.hypot(ax - state.x, ay - state.y)

    # ------------------------------------------------------------- tick

    def _tick(self) -> None:
        # Default: zero output. Anything that fails below leaves the car
        # commanding nothing rather than the previous tick's cached
        # response — fail-safe under SLAM/path dropout.
        cmd = ControlCommand()
        cmd.throttle = 0.0
        cmd.steering = 0.0
        cmd.brake = 0.0

        state = self._build_state()
        ref = ReferenceTrajectory.from_xy(
            self._latest_path_xs, self._latest_path_ys,
            kappa=self._latest_path_kappas or None,
        )

        if state is None or ref.empty:
            self._cmd_pub.publish(cmd)
            return

        # Accumulate travel distance — used by the stop-latch guard to
        # ignore the start gate's big-orange cones until we've driven a
        # full lap-ish away from spawn.
        if self._last_pose_xy is None:
            self._last_pose_xy = (state.x, state.y)
        else:
            dx = state.x - self._last_pose_xy[0]
            dy = state.y - self._last_pose_xy[1]
            self._travelled += math.hypot(dx, dy)
            self._last_pose_xy = (state.x, state.y)

        # Stop semantics — populated here so both controllers see the same
        # snapshot. Distance is to the latched orange-gate anchor in odom
        # frame; treated as +inf until ≥2 big-orange cones have been seen
        # at least once.
        ref.stop_distance = self._stop_distance(state)
        ref.stop_latched = self._stop_latched

        # Strategies own all algorithm logic. Per-tick state is immutable;
        # any internal accumulator (PI integral) lives on the strategy.
        steering_norm = self.lateral.compute(state, ref)
        throttle, regen = self.longitudinal.compute(state, ref)

        cmd.throttle = float(max(0.0, min(1.0, throttle)))
        cmd.brake = float(max(0.0, min(1.0, regen)))
        # Sign convention boundary. The strategy contract declares positive
        # steering = LEFT (math convention, matches Pure Pursuit's curvature
        # κ = 2·sin(α)/Ld where α is atan2(body_y, body_x) with body_y =
        # left). UE5 / Chaos SetSteeringInput uses the automotive convention:
        # positive = RIGHT (clockwise). Flip here so every controller can be
        # written in math without each one needing to know about the wire
        # convention.
        cmd.steering = float(max(-1.0, min(1.0, -steering_norm)))
        self._cmd_pub.publish(cmd)

        # Diagnostic publish (#260 follow-up). v_set vs SLAM v shows
        # whether the velocity controller is honouring corner-radius
        # braking; kappa_max shows whether it's even seeing the corner.
        # Both expose internal state that's otherwise only visible in
        # the per-tick log line — having them as topics lets Lichtblick
        # plot them against time alongside the SLAM speed trace.
        v_set_msg = Float32()
        v_set_msg.data = float(getattr(self.longitudinal, "last_v_set", 0.0))
        self._v_set_pub.publish(v_set_msg)
        kappa_msg = Float32()
        kappa_msg.data = float(getattr(self.longitudinal, "last_kappa_max", 0.0))
        self._kappa_max_pub.publish(kappa_msg)

        # Heartbeat — every ~0.5 s. Tells us at a glance whether each
        # stage is producing what we expect.
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count % 20 == 0:
            self.get_logger().info(
                f"v={state.speed:.2f} travelled={self._travelled:.1f}m -> "
                f"thr={cmd.throttle:+.3f} regen={cmd.brake:+.3f} "
                f"steer={cmd.steering:+.3f} | "
                f"path_n={len(ref.x)} path_len={ref.length:.1f}m "
                f"stop_d={ref.stop_distance:.1f} latched={ref.stop_latched}"
            )

    def _build_state(self) -> Optional[VehicleState]:
        if self._latest_odom is None:
            return None
        o = self._latest_odom
        # Pose in odom frame. Yaw via quat2euler.
        q = o.pose.pose.orientation
        roll, pitch, yaw = quat2euler([q.w, q.x, q.y, q.z])
        return VehicleState(
            x=o.pose.pose.position.x,
            y=o.pose.pose.position.y,
            yaw=yaw,
            vx=o.twist.twist.linear.x,
            vy=o.twist.twist.linear.y,
            yaw_rate=o.twist.twist.angular.z,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
