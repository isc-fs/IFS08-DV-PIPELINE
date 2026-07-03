"""
Car pipeline — on-vehicle build. mode_manager + mission_control +
management trio + autonomy lifecycle nodes (odometry through control),
wired onto the REAL car sensor/actuator surface instead of the IFSSIM
/fsds/* bridge.

What makes this the "car" build (vs sim_pipeline / full_pipeline):

  * autonomy_actions(profile="car") — IMU and LiDAR are remapped onto
    the uDV (/imu/data_raw) and Hesai (/lidar_points) topics; the
    sim-only ground-truth taps are dropped.
  * NO car-side adapter nodes. The uDV (a micro-ROS endpoint) is the
    mission_control peer directly, over the stock-typed interface in
    topic_contract.py: it publishes /assi/state + /ami/mission and its
    sensors (/imu/data_raw, /steering_angle in rad, /motor_rpm from the
    inverter), and consumes /dv/status + /ctrl/cmd (geometry_msgs/Twist)
    + /force_ebs (std_srvs/SetBool). mission_control reconciles the same
    surface here that sim_supervisor (the sim uDV emulator) provides in
    the sim — one identical mission_control in both worlds. The unit
    conversions + actuation scaling that the old car_sensor_bridge /
    car_supervisor did on the DVPC now live in uDV firmware. See
    docs/CAR_ADAPTATION.md.
  * No sim_supervisor (that's the sim's uDV emulator).

NO IFSSIM / foxglove / bag-recorder either — those are sim/dev concerns.

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
    # Real-car management layout: no sim_supervisor (the uDV plays that
    # role over the stock-typed interface; nothing to launch DVPC-side).
    actions += management_actions(include_sim_supervisor=False)
    # Autonomy wired onto the real-vehicle topic surface.
    actions += autonomy_actions(profile="car")

    return LaunchDescription(actions)
