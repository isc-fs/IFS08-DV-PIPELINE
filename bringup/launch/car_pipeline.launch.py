"""
Car pipeline — on-vehicle build. mode_manager + mission_control +
management trio + autonomy lifecycle nodes (odometry through control),
wired onto the REAL car sensor/actuator surface instead of the IFSSIM
/fsds/* bridge.

What makes this the "car" build (vs sim_pipeline / full_pipeline):

  * autonomy_actions(profile="car") — IMU and LiDAR are remapped onto
    the uDV (/imu/data_raw) and Hesai (/lidar_points) topics; the
    sim-only ground-truth taps are dropped.
  * car_sensor_bridge — converts the uDV's degrees steering sensor into
    the radians the EKF expects, and republishes the inverter wheel
    speed as /motor_rpm (see the firmware gap in docs/CAR_ADAPTATION.md).
  * car_supervisor — replaces sim_supervisor: it is the action client of
    mission_control_node driven by the uDV's /assi/state + /ami/mission,
    and relays the control output back to the uDV (/steering/cmd,
    /force_ebs). No sim_supervisor here.

NO IFSSIM / foxglove / bag-recorder either — those are sim/dev concerns.

Usage:
  ros2 launch bringup car_pipeline.launch.py
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

from bringup.launch_common import (
    autonomy_actions,
    management_actions,
    use_sim_time_params,
)


def generate_launch_description() -> LaunchDescription:
    actions: list = [
        # Real car: real sensors carry real stamps and there is no /clock.
        # Nodes must run on the wall clock, so use_sim_time defaults false.
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]
    # Real-car management layout: no sim_supervisor.
    actions += management_actions(include_sim_supervisor=False)
    # Autonomy wired onto the real-vehicle topic surface.
    actions += autonomy_actions(profile="car")

    # Sensor input adapters: steering deg→rad + inverter→/motor_rpm.
    actions.append(Node(
        package="car_sensor_bridge",
        executable="car_sensor_bridge_node",
        name="car_sensor_bridge",
        output="screen",
        parameters=use_sim_time_params(),
    ))

    # Mission/actuation adapter: replaces sim_supervisor on the car.
    actions.append(Node(
        package="car_supervisor",
        executable="car_supervisor_node",
        name="car_supervisor",
        output="screen",
        parameters=use_sim_time_params(),
    ))

    return LaunchDescription(actions)
