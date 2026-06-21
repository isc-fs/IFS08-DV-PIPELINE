"""
Sim pipeline — management trio + autonomy lifecycle nodes
nodes. NO UE5 bridge, NO foxglove, NO bag recorder.

For developers running the autonomy locally against an
externally-launched simulator (FSDS, a bag replay, or a standalone
IFSSIM bridge started in another terminal). The full
docker-stack equivalent is full_pipeline.launch.py.

Usage:
  ros2 launch bringup sim_pipeline.launch.py
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
        # Launch args kept for parity with full_pipeline so a launcher
        # script can pass the same kwargs to either file. They're
        # accepted but unused here — the bridge that consumed them
        # isn't included in this layout.
        DeclareLaunchArgument("mission_name", default_value="trackdrive"),
        DeclareLaunchArgument("track_name",   default_value="A"),
        # Sim runs on the bridge's /clock (UE sim time). The externally-
        # launched bridge must be publishing /clock for the nodes to tick.
        DeclareLaunchArgument("use_sim_time",  default_value="true"),
    ]

    # Sim layout still includes sim_supervisor: even without the
    # bridge in this launch file, the supervisor is part of the
    # command-relay chain (control_node → mission_control → supervisor
    # → bridge subscriber). An externally-launched bridge picks up
    # /control_command from the supervisor's remap, same as in
    # full_pipeline.
    actions += management_actions(include_sim_supervisor=True)
    actions += autonomy_actions()

    return LaunchDescription(actions)
