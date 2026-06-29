"""
car_sensor_bridge_node — on-vehicle sensor input adapter.

Sits between the uDV / inverter and the autonomy pipeline, providing the
two EKF inputs that a plain topic remap cannot (because they need a unit
conversion or have no car source at all):

  1. Steering. Subscribes the uDV's /steering/angle_sensor (Float32,
     DEGREES) and republishes /steering_angle (Float32, RADIANS) — the
     unit odometry_filter_node / slam_node actually consume.

  2. Wheel speed. The uDV publishes NO motor-RPM topic. The car's source
     is the inverter; this node subscribes a (configurable) inverter
     feed and republishes /motor_rpm (Float32, motor-shaft RPM) for the
     EKF, which applies its own ekf.rpm_to_ms scaling.

⚠️  INVERTER SOURCE IS PROVISIONAL.  The LattePanda has no SocketCAN
    interface today, so the inverter feed must arrive as a ROS topic
    (e.g. relayed through the uDV or a future USB-CAN bridge). The input
    topic name, message type and units (eRPM vs mechanical RPM, sign,
    LSB) are NOT yet confirmed. They are fully parameterised below and
    flagged in docs/CAR_ADAPTATION.md so this is a one-line wiring
    change once the real feed exists — not a silent gap.

All heavy lifting is in conversions.py (pure, unit-tested). This node is
just rclpy plumbing + input validation.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float32

from car_sensor_bridge.conversions import (
    inverter_value_to_motor_rpm,
    steering_sensor_deg_to_rad,
)


# Best-effort, shallow depth — matches the uDV's best-effort sensor
# publishers and the EKF's BEST_EFFORT subscriptions. Latest-sample-wins
# is correct for a continuous sensor stream.
_SENSOR_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


class CarSensorBridge(Node):
    """See module docstring."""

    NODE_NAME = "car_sensor_bridge"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # --- Steering parameters (deg → rad) ---
        self._steering_in = self.declare_parameter(
            "steering_in_topic", "/steering/angle_sensor").value
        self._steering_out = self.declare_parameter(
            "steering_out_topic", "/steering_angle").value
        # 1.0 ⇒ sensor already measures the road-wheel angle. Set to the
        # steering ratio if it measures the wheel/column angle. CONFIRM.
        self._steering_ratio = float(self.declare_parameter(
            "steering_ratio", 1.0).value)
        self._steering_sign = float(self.declare_parameter(
            "steering_sign", 1.0).value)
        self._steering_offset_deg = float(self.declare_parameter(
            "steering_offset_deg", 0.0).value)

        # --- Inverter → motor RPM parameters ---
        # Default input topic is a placeholder; nothing publishes it yet.
        self._inverter_in = self.declare_parameter(
            "inverter_in_topic", "/inverter/erpm").value
        self._motor_rpm_out = self.declare_parameter(
            "motor_rpm_out_topic", "/motor_rpm").value
        self._inverter_is_erpm = bool(self.declare_parameter(
            "inverter_is_erpm", True).value)
        self._pole_pairs = int(self.declare_parameter(
            "pole_pairs", 10).value)
        self._inverter_scale = float(self.declare_parameter(
            "inverter_scale", 1.0).value)

        # --- Publishers ---
        self._steering_pub = self.create_publisher(
            Float32, self._steering_out, _SENSOR_QOS)
        self._motor_rpm_pub = self.create_publisher(
            Float32, self._motor_rpm_out, _SENSOR_QOS)

        # --- Subscriptions ---
        self.create_subscription(
            Float32, self._steering_in, self._on_steering, _SENSOR_QOS)
        self.create_subscription(
            Float32, self._inverter_in, self._on_inverter, _SENSOR_QOS)

        # Throttle the "no inverter data yet" warning so a missing feed
        # doesn't spam the log; only counts received samples for info.
        self._inverter_samples = 0

        self.get_logger().info(
            f"car_sensor_bridge up: steering {self._steering_in} (deg) → "
            f"{self._steering_out} (rad) "
            f"[ratio={self._steering_ratio}, sign={self._steering_sign}, "
            f"offset={self._steering_offset_deg}deg]; "
            f"inverter {self._inverter_in} → {self._motor_rpm_out} "
            f"[is_erpm={self._inverter_is_erpm}, "
            f"pole_pairs={self._pole_pairs}, scale={self._inverter_scale}]")
        self.get_logger().warn(
            "INVERTER /motor_rpm SOURCE IS PROVISIONAL — input topic "
            f"{self._inverter_in!r} is a placeholder. Confirm the real "
            "inverter feed (topic/type/units) before on-track running; "
            "until then the EKF runs IMU+steering only (see "
            "docs/CAR_ADAPTATION.md).")

    # ------------------------------------------------------------------
    def _on_steering(self, msg: Float32) -> None:
        if not math.isfinite(msg.data):
            self.get_logger().warn(
                f"dropping non-finite steering sample {msg.data!r}")
            return
        try:
            rad = steering_sensor_deg_to_rad(
                float(msg.data),
                steering_ratio=self._steering_ratio,
                sign=self._steering_sign,
                offset_deg=self._steering_offset_deg,
            )
        except ValueError as ex:
            self.get_logger().warn(f"steering conversion skipped: {ex}")
            return
        self._steering_pub.publish(Float32(data=float(rad)))

    # ------------------------------------------------------------------
    def _on_inverter(self, msg: Float32) -> None:
        if not math.isfinite(msg.data):
            self.get_logger().warn(
                f"dropping non-finite inverter sample {msg.data!r}")
            return
        try:
            rpm = inverter_value_to_motor_rpm(
                float(msg.data),
                is_erpm=self._inverter_is_erpm,
                pole_pairs=self._pole_pairs,
                scale=self._inverter_scale,
            )
        except ValueError as ex:
            self.get_logger().warn(f"inverter conversion skipped: {ex}")
            return
        self._motor_rpm_pub.publish(Float32(data=float(rpm)))
        self._inverter_samples += 1
        if self._inverter_samples == 1:
            self.get_logger().info(
                f"first inverter sample received on {self._inverter_in} — "
                f"/motor_rpm is now live")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarSensorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
