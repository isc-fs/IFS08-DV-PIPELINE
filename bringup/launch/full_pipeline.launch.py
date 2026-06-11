"""
Full pipeline — UE5 bridge + foxglove + bag recorder + management
trio (mode_manager / mission_control / sim_supervisor) +
management trio + autonomy lifecycle nodes (odometry through control).

This is what docker/dv_pipeline_stack runs at container startup. The
docker launch file is now a thin wrapper that includes this one with
the same launch arguments forwarded through.

Usage:

  ros2 launch bringup full_pipeline.launch.py
  ros2 launch bringup full_pipeline.launch.py mission_name:=autocross

For sim development without the IFSSIM bridge (e.g. against a bag
replay or an externally-launched FSDS), use sim_pipeline.launch.py.
For the on-vehicle build, use car_pipeline.launch.py.
"""
from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from bringup.launch_common import (
    autonomy_actions,
    management_actions,
)


# Opt-in /lidar/Lidar1/viz subsampling for browser-based visualisers.
# 0 (default) = disabled; >=2 = every-Nth-point cloud alongside the
# full /lidar/Lidar1 feed. Autonomy stack always subscribes to the
# full cloud.
try:
    LIDAR_VIZ_DECIMATION = int(os.environ.get("LIDAR_VIZ_DECIMATION", "0"))
except ValueError:
    LIDAR_VIZ_DECIMATION = 0


# foxglove_bridge topic whitelist. Subscribing to all 49+ topics on the
# graph idled the bridge at 53 % CPU before any client connected; this
# list is the union of topics referenced across the four
# /lichtblick/*.json dashboards (auto-extracted 2026-05-11). Anything
# new a dashboard needs has to land here too.
_FOXGLOVE_TOPIC_WHITELIST = [
    # Cone perception + map
    "/Conos", "/Conos_Orange", "/Conos_raw",
    # Planner output + debug
    "/Path", "/path_planning/debug",
    # SLAM
    "/slam/pose", "/cone_slam/gt_aligned",
    "/cone_slam/gt_error_m",
    # Controller diagnostics
    "/control/v_set_mps", "/control/kappa_max_per_m",
    "/ctrl/cmd_internal",
    # Sensors actually plotted (no full LiDAR — viz only)
    "/lidar/Lidar1/viz", "/imu", "/motor_rpm",
    # Diagnostic GT
    "/testing_only/odom", "/testing_only/track",
    # Lichtblick built-ins (interactive feature topics)
    "/clicked_point", "/initialpose",
    "/move_base_simple/goal", "/track_overlay",
    # TF is mandatory for the 3D panel
    "/tf", "/tf_static",
]


def generate_launch_description() -> LaunchDescription:
    actions: list = [
        # ------------------ Launch arguments ------------------
        DeclareLaunchArgument("host",         default_value="host.docker.internal"),
        DeclareLaunchArgument("port",         default_value="41451"),
        DeclareLaunchArgument("mission_name", default_value="trackdrive"),
        DeclareLaunchArgument("track_name",   default_value="A"),

        # ------------------ UE5 bridge ------------------
        Node(
            package="ifssim_bridge",
            executable="ifssim_bridge",
            name="ifssim_bridge",
            output="screen",
            parameters=[{
                "host":                 LaunchConfiguration("host"),
                "port":                 LaunchConfiguration("port"),
                "mission_name":         LaunchConfiguration("mission_name"),
                "track_name":           LaunchConfiguration("track_name"),
                "competition_mode":     False,
                "lidar_viz_decimation": LIDAR_VIZ_DECIMATION,
            }],
        ),

        # ------------------ Foxglove bridge ------------------
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port":              8765,
                "address":           "0.0.0.0",
                "send_buffer_limit": 64 * 1024 * 1024,
                "use_sim_time":      False,
                "topic_whitelist":   _FOXGLOVE_TOPIC_WHITELIST,
                "use_compression":   True,
            }],
        ),

        # ------------------ Bag recorder ------------------
        # Always-on plain Node hosting /bag_recorder/start +
        # /bag_recorder/stop services. The MC web backend's
        # bag_recorder.py is the client; the actual `ros2 bag record`
        # subprocess runs HERE (inside dv_pipeline_stack) so it shares
        # the SHM-tuned DDS context with the publishers — full-fidelity
        # 10 Hz LiDAR capture, no UDP-fragmentation drops.
        Node(
            package="bag_recorder_node",
            executable="bag_recorder_node",
            name="bag_recorder_node",
            output="screen",
        ),
    ]

    # ------------------ Management + autonomy ------------------
    actions += management_actions(include_sim_supervisor=True)
    actions += autonomy_actions()

    return LaunchDescription(actions)
