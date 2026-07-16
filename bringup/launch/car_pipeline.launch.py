"""
Car pipeline — on-vehicle build. mode_manager + mission_control +
management trio + autonomy lifecycle nodes (odometry through control),
wired onto the REAL car sensor/actuator surface instead of the IFSSIM
/fsds/* bridge.

What makes this the "car" build (vs sim_pipeline / full_pipeline):

  * autonomy_actions(profile="car") — only the LiDAR is remapped onto the
    Hesai (/lidar_points) topic; IMU is canonical /imu on both sides (no
    remap); the sim-only ground-truth taps are dropped.
  * NO car-side adapter nodes. The uDV (a micro-ROS endpoint) is the
    mission_control peer directly, over the stock-typed interface in
    topic_contract.py: it publishes /assi/state + /ami/mission and its
    sensors (/imu, /steering_angle in rad, /motor_rpm from the
    inverter), and consumes /dv/status + /ctrl/cmd (geometry_msgs/Twist)
    + /force_ebs (std_srvs/SetBool). mission_control reconciles the same
    surface here that sim_supervisor (the sim uDV emulator) provides in
    the sim — one identical mission_control in both worlds. The unit
    conversions + actuation scaling that the old car_sensor_bridge /
    car_supervisor did on the DVPC now live in uDV firmware. See
    docs/CAR_ADAPTATION.md.
  * No sim_supervisor (that's the sim's uDV emulator).

No IFSSIM / foxglove here — those are sim/dev concerns. The bag recorder,
by contrast, IS launched on the car when free_run is on: the always-on
data-collection floor records every powered-on session (see free_run below).

## free_run

`free_run` (default true) turns on the always-on autonomy floor: while the
uDV is powered on (heartbeat alive) mission_control brings the WHOLE autonomy
stack up — perception, SLAM, planning AND control — and records a rosbag,
even in AS OFF / manual driving with the ASMS unpowered, so data is captured
without ever arming the car. control_node runs so its would-be commands land
on /ctrl/cmd_internal (recorded) for pilot-vs-autonomy comparison, but the
pipeline does NOT relay them: /ctrl/cmd is published only in a live run, and
the uDV actuates only in AS Driving. At the go edge control_node is clean-
reset for the run (fresh state, no SLAM reset / Numba re-JIT). Flip it off
for a lighter-CPU competition build:

    ros2 launch bringup car_pipeline.launch.py free_run:=false

Usage:
  ros2 launch bringup car_pipeline.launch.py
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from bringup.launch_common import (
    autonomy_actions,
    management_actions,
)


def generate_launch_description() -> LaunchDescription:
    actions: list = [
        # Real car: real sensors carry real stamps and there is no /clock.
        # Nodes must run on the wall clock, so use_sim_time defaults false.
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        # Always-on data-collection floor (see module docstring). Default on
        # for now; flip to false for a lighter-CPU build.
        DeclareLaunchArgument("free_run", default_value="true"),
        # ASB hard stop at mission end (DV_STOPPING). SAFETY-CRITICAL, so it is
        # a launch arg rather than a source default: it stays FALSE for every
        # normal run and is only ever set true for a bench-validated run with
        # byte-7 firmware flashed:
        #     ros2 launch bringup car_pipeline.launch.py hard_stop_on_finish:=true
        # See docs/HARD_STOP_BENCH.md for the three-stage validation that must
        # pass BEFORE this is enabled on a real run.
        DeclareLaunchArgument("hard_stop_on_finish", default_value="false"),

        # ------------------ Bag recorder ------------------
        # Always-on service host (/bag_recorder/start + /bag_recorder/stop);
        # the `ros2 bag record` subprocess runs HERE, in the pipeline's DDS
        # context, for full-fidelity capture. mission_control auto-drives it
        # while free_run is active. (Same node the sim's full_pipeline runs.)
        Node(
            package="bag_recorder_node",
            executable="bag_recorder_node",
            name="bag_recorder_node",
            output="screen",
        ),
    ]
    # Real-car management layout: no sim_supervisor (the uDV plays that
    # role over the stock-typed interface; nothing to launch DVPC-side).
    actions += management_actions(
        include_sim_supervisor=False,
        free_run=LaunchConfiguration("free_run"),
        hard_stop_on_finish=LaunchConfiguration("hard_stop_on_finish"),
    )
    # Autonomy wired onto the real-vehicle topic surface.
    actions += autonomy_actions(profile="car")

    return LaunchDescription(actions)
