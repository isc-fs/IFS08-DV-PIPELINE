#!/usr/bin/env python3
"""prerun re-stamp relay — republish bagged sensor feeds with `now` stamps.

For the ``prerun/`` branch (see PRERUN.md): we replay a recorded rosbag ON the
car to exercise the live autonomy + actuation. The bag's messages carry stamps
from the original recording, but the live DV handshake and the 400 ms
staleness watchdogs run on *current* time — so SLAM/odometry time logic and
the watchdogs would misbehave on old stamps. This node subscribes to the bag's
feeds on shadow topics and republishes them to the real topics with
``header.stamp`` rewritten to ``now``.

Only ``/imu`` (sensor_msgs/Imu) and ``/lidar_points`` (sensor_msgs/PointCloud2)
are relayed — the two feeds suppressed on the uDV/pipeline prerun branches.
Any other topic in the bag is NOT relayed and would collide with the live
system: strip it from the bag or add a stub.

Usage (on the DVPC, workspace sourced):

    # 1) pipeline up (prerun branch: Hesai driver off; autonomy consumes
    #    /imu + /lidar_points). uDV flashed from its prerun branch (/imu off).
    # 2) this relay:
    python3 deploy/prerun_restamp_relay.py
    # 3) play the bag into the shadow topics (1x so timing is preserved):
    ros2 bag play <bag> --remap /imu:=/imu_bag /lidar_points:=/lidar_points_bag

Topic names are overridable as ROS params (imu_in/imu_out/lidar_in/lidar_out).
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2


class RestampRelay(Node):
    def __init__(self) -> None:
        super().__init__("prerun_restamp_relay")
        imu_in = self.declare_parameter("imu_in", "/imu_bag").value
        imu_out = self.declare_parameter("imu_out", "/imu").value
        lidar_in = self.declare_parameter("lidar_in", "/lidar_points_bag").value
        lidar_out = self.declare_parameter("lidar_out", "/lidar_points").value

        # sensor_data QoS (best-effort) matches the firmware /imu publisher and
        # the SLAM / cone_detection sensor subscriptions.
        self._imu_pub = self.create_publisher(Imu, imu_out, qos_profile_sensor_data)
        self._lidar_pub = self.create_publisher(
            PointCloud2, lidar_out, qos_profile_sensor_data)
        self.create_subscription(
            Imu, imu_in, self._relay_imu, qos_profile_sensor_data)
        self.create_subscription(
            PointCloud2, lidar_in, self._relay_lidar, qos_profile_sensor_data)

        self._imu_n = 0
        self._lidar_n = 0
        self.create_timer(2.0, self._log_stats)
        self.get_logger().info(
            f"prerun re-stamp relay up: {imu_in}->{imu_out}, "
            f"{lidar_in}->{lidar_out} (header.stamp rewritten to now)")

    def _relay_imu(self, msg: Imu) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        self._imu_pub.publish(msg)
        self._imu_n += 1

    def _relay_lidar(self, msg: PointCloud2) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        self._lidar_pub.publish(msg)
        self._lidar_n += 1

    def _log_stats(self) -> None:
        self.get_logger().info(f"relayed imu={self._imu_n} lidar={self._lidar_n}")


def main() -> None:
    rclpy.init()
    node = RestampRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
