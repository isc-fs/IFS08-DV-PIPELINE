#!/usr/bin/env bash
# =============================================================================
# dv_manual.sh — record a rosbag during MANUAL driving, WITHOUT the pipeline.
#
# Started by dv-manual.service (`dv manual`). Brings up ONLY the sensor layer
# (Hesai LiDAR driver + static TFs, via car_sensors.launch.py) — NOT
# mission_control, NOT the autonomy lifecycle nodes — then records the full
# sensor graph, exactly like the race recorder:
#
#   /lidar_points  raw cloud     ← Hesai driver (started here)
#   /imu /motor_rpm /steering_angle  ← the uDV, over microros-agent.service
#                                      (independent of this; present if it's up)
#
# Why this exists: DV testing time is scarce, so we want a bag from every run
# including manual laps (offline perception/SLAM/odom development replays the
# manual bag through the pipeline later). The race recorder is bolted to
# dv-pipeline.service and waits for /dv/status, so it can't record manual
# driving. This is the manual-mode twin.
#
# Lifecycle: the sensor launch runs in the background inside this unit's cgroup;
# the recorder runs in the foreground (exec). `dv stop` / `systemctl stop
# dv-manual.service` sends SIGINT to the whole cgroup (KillMode=control-group),
# so rosbag2 finalizes the bag AND the sensor nodes shut down cleanly.
#
# Bags land in the same place as race bags (/home/isc/bags) with a `manual_`
# prefix. Honours /etc/dv/norecord. Same disk floor + best-effort QoS handling
# as dv_record.sh (this reuses it).
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh).
# =============================================================================
set -o pipefail
LOG(){ logger -t dv-manual "$*" 2>/dev/null; echo "[dv-manual] $*"; }

if [ -f /etc/dv/norecord ]; then
  LOG "disabled by /etc/dv/norecord — not recording."
  exit 0
fi

source /opt/ros/humble/setup.bash 2>/dev/null
source /home/isc/ros2_ws/install/local_setup.bash 2>/dev/null
source /home/isc/dv_ws/install/local_setup.bash 2>/dev/null
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

WITH_LIDAR="${DV_MANUAL_WITH_LIDAR:-true}"

# Bring up the sensor layer in the background (same cgroup, so it dies with the
# unit). No autonomy: this is car_sensors, not car_bringup.
LOG "starting sensor layer (car_sensors.launch.py, with_lidar=${WITH_LIDAR})…"
ros2 launch bringup car_sensors.launch.py with_lidar:="${WITH_LIDAR}" &
SENSORS_PID=$!

# If the sensor launch dies, don't leave a recorder running against a dead
# graph — take the whole unit down.
trap 'kill "$SENSORS_PID" 2>/dev/null' EXIT

# Hand off to the shared record core, gated on the sensor topic (best-effort →
# method=exists, not echo) and labelled "manual". The recorder execs
# `ros2 bag record`, replacing this shell; the backgrounded sensor launch stays
# in the cgroup and is reaped by systemd's control-group kill on stop.
WARMUP_TOPIC="/lidar_points"
[ "$WITH_LIDAR" = "true" ] || WARMUP_TOPIC="/imu"   # no lidar → wait on the IMU

export DV_RECORD_WARMUP_TOPIC="$WARMUP_TOPIC"
export DV_RECORD_WARMUP_METHOD="exists"
export DV_RECORD_MODE_LABEL="manual"
# The IMU/RPM/steering come from the uDV best-effort too; force best-effort
# readers on them as well as the cloud so a manual bag isn't silently empty.
export DV_RECORD_BEST_EFFORT="${DV_RECORD_BEST_EFFORT-/lidar_points /imu /motor_rpm /steering_angle}"

LOG "handing off to dv_record.sh (warmup ${WARMUP_TOPIC} exists, label manual)…"
exec /usr/local/bin/dv_record.sh
