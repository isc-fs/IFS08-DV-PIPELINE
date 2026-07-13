#!/usr/bin/env bash
# =============================================================================
# dv_record.sh — race telemetry rosbag recorder (run by dv-record.service).
#
# Started automatically whenever dv-pipeline.service starts (race boot,
# `dv race`, `dv restart`) via the unit's WantedBy=dv-pipeline.service;
# PartOf= stops it with the pipeline. Waits for the pipeline warmup —
# /dv/status is published by mission_control once the stack is up, so its
# appearance means the topic graph exists — then records every topic except
# the raw lidar cloud (4.5 MB × 10 Hz ≈ 45 MB/s; cone/SLAM/path outputs are
# recorded and are what post-run analysis uses).
#
# Knobs (systemd drop-in or environment):
#   DV_RECORD_DIR             output dir            (default /home/isc/bags)
#   DV_RECORD_EXCLUDE         record -x regex       (default ^/lidar_points$;
#                             set empty to record everything)
#   DV_RECORD_WARMUP_TIMEOUT  seconds to wait for /dv/status  (default 90)
#   DV_RECORD_MIN_FREE_GB     refuse to record below this free space (default 10)
# Opt-out: `touch /etc/dv/norecord` (isc-owned dir) disables recording.
#
# The unit sends SIGINT on stop (KillSignal) so rosbag2 finalizes
# metadata.yaml — a hard kill leaves a corrupt bag needing
# `rm metadata.yaml && ros2 bag reindex <bag> -s mcap`.
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh).
# =============================================================================
# No `set -u`: sourcing ROS setup.bash references unbound vars.
set -o pipefail
LOG(){ logger -t dv-record "$*" 2>/dev/null; echo "[dv-record] $*"; }

if [ -f /etc/dv/norecord ]; then
  LOG "disabled by /etc/dv/norecord — not recording."
  exit 0
fi

source /opt/ros/humble/setup.bash 2>/dev/null
source /home/isc/ros2_ws/install/local_setup.bash 2>/dev/null
source /home/isc/dv_ws/install/local_setup.bash 2>/dev/null
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

OUTDIR="${DV_RECORD_DIR:-/home/isc/bags}"
EXCLUDE="${DV_RECORD_EXCLUDE-^/lidar_points$}"
WARMUP_TIMEOUT="${DV_RECORD_WARMUP_TIMEOUT:-90}"
MIN_FREE_GB="${DV_RECORD_MIN_FREE_GB:-10}"

mkdir -p "$OUTDIR"

free_gb=$(( $(df -Pk "$OUTDIR" | awk 'NR==2{print $4}') / 1024 / 1024 ))
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  LOG "only ${free_gb} GB free in $OUTDIR (< ${MIN_FREE_GB}) — refusing to record."
  exit 1
fi

# Warmup: wait for mission_control's /dv/status heartbeat (the last node up).
LOG "waiting up to ${WARMUP_TIMEOUT}s for pipeline warmup (/dv/status)…"
up=0
for _ in $(seq 1 "$WARMUP_TIMEOUT"); do
  if timeout 3 ros2 topic echo --once /dv/status >/dev/null 2>&1; then up=1; break; fi
  sleep 1
done
if [ "$up" != 1 ]; then
  LOG "pipeline never warmed up (/dv/status silent after ${WARMUP_TIMEOUT}s) — giving up."
  exit 1
fi
sleep 2   # let the rest of the topic graph register with discovery

MODE=$(cat /run/dv_mode 2>/dev/null || echo run)
OUT="$OUTDIR/${MODE}_$(date +%Y%m%d_%H%M%S)"
if [ -n "$EXCLUDE" ]; then
  LOG "recording -> $OUT (excluding '$EXCLUDE')"
  exec ros2 bag record -s mcap -o "$OUT" -a -x "$EXCLUDE"
else
  LOG "recording -> $OUT (all topics)"
  exec ros2 bag record -s mcap -o "$OUT" -a
fi
