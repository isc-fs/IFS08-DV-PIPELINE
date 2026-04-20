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

    def _setup_subscribers(self) -> None:
        """Subscribe to the topics providing path, velocity, and odometry feedback."""
        self.path_subscriber = self.create_subscription(
            Path, "Path", self.path_callback, self.QUEUE_SIZE
        )
        self.velocity_subscriber = self.create_subscription(
            TwistWithCovarianceStamped,
            "/fsds/gss",
            self.velocity_callback,
            self.QUEUE_SIZE,
        )
        self.odom_subscriber = self.create_subscription(
            Odometry,
            "/fsds/testing_only/odom",
            self.odometry_callback,
            self.QUEUE_SIZE,
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
        # Margin past the orange gate centroid into the Stop Area. FS rules
        # require the car to stop at least 3 m after the finish line.
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
        #   steering_damp_gain: doubled to smooth high-frequency oscillation
        #     observed on straights.
        self.declare_parameter("control_gain", 2.5)
        self.declare_parameter("softening_gain", 6.0)
        self.declare_parameter("yaw_rate_gain", 0.0)
        self.declare_parameter("steering_damp_gain", 1.0)
        # Wheelbase for the Stanley front-axle projection. FSG T 8 requires
        # >= 1525 mm, typical designs sit around 1.5–1.7 m. Was hardcoded to
        # 3.0 m via the controller's Python default — completely wrong and
        # skewed every cross-track calculation. Now explicit and tunable.
        self.declare_parameter("wheelbase", 1.55)

        # Max steering angle
        self.declare_parameter("max_steering_ang", 25.0)

        # Velocity control parameters
        self.declare_parameter("Kp_vel", 0.05)
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

        # Feed the local path into the Stanley controller and obtain the steering demand
        self.stanley_controller.set_path(px, py, pyaw)
        (limited_steering_angle, _target_index, crosstrack_error) = (
            self.stanley_controller.stanley_control(
                float(odom_to_vehicle.transform.translation.x),
                float(odom_to_vehicle.transform.translation.y),
                float(yaw),
                self.velocity,
                self.steering,
            )
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
        distance_to_path_end = self._compute_orange_stop_distance()
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

        # If we're at or past the latched stop target, engage EBS: publish
        # one /signal/ebs message, then the bridge applies
        # setCarControls 0 0 1 + disableApiControl on UE5 (equivalent to
        # real-car EBS). Any subsequent setCarControls is silently dropped
        # by the bridge, so the brake stays latched for good regardless of
        # what we publish next. Still emit brake=1 / throttle=0 locally as
        # a safety net for the tick or two before disableApiControl lands.
        if (
            self.latched_stop_traveled is not None
            and self.distance_traveled >= self.latched_stop_traveled - 1.0
        ):
            command_msg.throttle = 0.0
            command_msg.brake = 1.0
            if not self.ebs_requested:
                self.ebs_publisher.publish(Empty())
                self.ebs_requested = True
                self.get_logger().info("EBS requested — autonomous stop engaged")

        # Gate steering: don't apply Stanley steering until the car is moving.
        # At near-zero speed the cross-track term saturates to max lock, causing
        # the car to spin before it has any forward momentum.
        if abs(self.velocity) < 0.5:
            self.steering = 0.0
        else:
            self.steering = -limited_steering_angle
        command_msg.steering = self.steering / np.deg2rad(self.max_steering_ang)

        self.command_publisher.publish(command_msg)

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
