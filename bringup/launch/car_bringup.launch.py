"""
Car bringup — the dv-pipeline.service entry point: sensors + autonomy.

Composed of two layers, each launchable on its own:

  * car_sensors.launch.py  — Hesai ATX LiDAR driver (/lidar_points) + static
    base_link→sensor TFs + optional foxglove. Runs standalone as the manual-
    driving recording layer (dv-manual.service); see deploy/dv_manual.sh.
  * car_pipeline.launch.py — autonomy only (mode_manager + mission_control +
    lifecycle nodes on the real-vehicle topic surface).

This wrapper is just the two together, replacing the old isc_ws/isc_startup
stub stack as the DVPC race-mode entry point. The sensor geometry / TF details
live in car_sensors.launch.py now.

Usage (dv-pipeline.service / bench):
  ros2 launch bringup car_bringup.launch.py
  ros2 launch bringup car_bringup.launch.py foxglove:=true      # bench
  ros2 launch bringup car_bringup.launch.py with_lidar:=false   # no ATX
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    with_lidar = LaunchConfiguration("with_lidar")
    foxglove = LaunchConfiguration("foxglove")

    def _bringup_launch(name: str):
        return PathJoinSubstitution([
            FindPackageShare("bringup"), "launch", name])

    return LaunchDescription([
        DeclareLaunchArgument(
            "with_lidar", default_value="true",
            description="Start the Hesai ATX driver (needs hesai_ros_driver "
                        "from the DVPC's ros2_ws sourced)."),
        DeclareLaunchArgument(
            "foxglove", default_value="false",
            description="Start foxglove_bridge (bench/umbilical monitoring)."),

        # --- Sensor layer (Hesai + static TFs + optional foxglove) ---------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(_bringup_launch("car_sensors.launch.py")),
            launch_arguments={
                "with_lidar": with_lidar,
                "foxglove": foxglove,
            }.items(),
        ),

        # --- Autonomy (management + lifecycle nodes, car profile) ----------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(_bringup_launch("car_pipeline.launch.py")),
        ),
    ])
