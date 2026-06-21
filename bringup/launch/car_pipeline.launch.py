"""
Car pipeline — on-vehicle build. mode_manager + mission_control +
management trio + autonomy lifecycle nodes (odometry through control).

NO sim_supervisor (the on-vehicle uDV microcontroller replaces it as
the bridge between the autonomy and the actuators). NO IFSSIM /
foxglove / bag-recorder either — those are sim/dev concerns.

The autonomy still expects to find the same topics it consumes in
sim: /imu, /motor_rpm, /lidar/Lidar1, /odom, /testing_only/odom (the
last one only if a reference is available). On the car these are
provided by the uDV ROS bridge / LiDAR driver / VESC bridge running
in separate processes.

Usage:
  ros2 launch bringup car_pipeline.launch.py
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument

from bringup.launch_common import (
    autonomy_actions,
    management_actions,
)


def generate_launch_description() -> LaunchDescription:
    actions: list = [
        # Real car: real sensors carry real stamps and there is no /clock.
        # Nodes must run on the wall clock, so use_sim_time defaults false.
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]
    # Real-car management layout: no sim_supervisor.
    actions += management_actions(include_sim_supervisor=False)
    actions += autonomy_actions()
    return LaunchDescription(actions)
