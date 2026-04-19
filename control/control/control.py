from math import cos, sin
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from fs_msgs.msg import ControlCommand, FinishedSignal
from geometry_msgs.msg import TransformStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
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
        self.finished_subscriber = self.create_subscription(
            FinishedSignal,
            "/signal/finished",
            self.finished_callback,
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
        # Once the referee declares the event finished we latch full brake and
        # ignore subsequent path updates; the latch is cleared only when the
        # node is restarted (which happens every new Start Session).
        self.finished: bool = False

    def _setup_timer(self) -> None:
        """Schedule the periodic execution of the control loop callback."""
        self.timer = self.create_timer(
            1.0 / self.PUBLISH_RATE_HZ, self.control_loop_callback
        )

    def _init_params(self) -> None:
        """Declare the ROS parameters that configure the controller gains."""

        # Stanley controller parameters
        self.declare_parameter("control_gain", 4.0)
        self.declare_parameter("softening_gain", 6.0)
        self.declare_parameter("yaw_rate_gain", 0.0)
        self.declare_parameter("steering_damp_gain", 0.5)
        # TODO: check wheelbase param

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

    def finished_callback(self, msg: FinishedSignal) -> None:
        """Latch the finished flag once the referee ends the event."""
        if not self.finished:
            self.get_logger().info("Event finished — latching full brake")
        self.finished = True

    def control_loop_callback(self) -> None:
        """Run the primary control algorithm responsible for steering and speed."""
        command_msg = ControlCommand()

        # After the event finishes, hold the car at full brake. This runs
        # before the transform/path lookups so control keeps braking even if
        # SLAM or path planning drift after the finish gate.
        if self.finished:
            command_msg.brake = 1.0
            command_msg.throttle = 0.0
            command_msg.steering = 0.0
            self.command_publisher.publish(command_msg)
            return

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
        # Compute the required longitudinal command given the target profile
        velocity_command_value, velocity_ref = (
            self.velocity_controller.get_control_value(
                self.velocity, forward_path_points, crosstrack_error
            )
        )

        # Acceleration and braking are limited inside the velocity controller
        if velocity_command_value > 0:
            command_msg.throttle = velocity_command_value
        else:
            command_msg.brake = -velocity_command_value

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
