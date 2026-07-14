#!/usr/bin/env bash
# =============================================================================
# dv_record.sh — race rosbag recorder (run by dv-record.service).
#
# Started automatically whenever dv-pipeline.service starts (race boot,
# `dv race`, `dv restart`) via the unit's WantedBy=dv-pipeline.service;
# PartOf= stops it with the pipeline. Waits for the pipeline warmup —
# /dv/status is published by mission_control once the stack is up, so its
# appearance means the topic graph exists — then records EVERY topic,
# INCLUDING the raw lidar cloud (/lidar_points), so a run can be replayed and
# you can see exactly what the lidar saw.
#
# ⚠ THROUGHPUT: /lidar_points is ~4.5 MB × 10 Hz ≈ 45 MB/s ≈ 2.7 GB/min. A 5-min
# run is ~13 GB. That is deliberate — the cloud is the one signal that lets us
# debug perception offline (point_step/layout, why a cone was or wasn't
# detected, residual histograms). Set DV_RECORD_EXCLUDE='^/lidar_points$' to go
# back to the small telemetry-only bag if disk gets tight.
#
# ⚠ QoS (why the override below is NOT optional): the Hesai driver publishes
# /lidar_points BEST_EFFORT (SensorDataQoS). A RELIABLE subscriber is
# QoS-INCOMPATIBLE with a best-effort publisher and receives NOTHING — so
# recording the topic without forcing a best-effort reader silently produces a
# /lidar_points topic with ZERO messages. A best-effort reader matches both
# best-effort and reliable publishers, so the override is always safe.
#
# Knobs (systemd drop-in or environment):
#   DV_RECORD_DIR             output dir            (default /home/isc/bags)
#   DV_RECORD_EXCLUDE         record -x regex       (default EMPTY = record all,
#                             incl. /lidar_points). e.g. '^/lidar_points$'
#   DV_RECORD_BEST_EFFORT     space-separated topics to force a best-effort
#                             reader on          (default "/lidar_points")
#   DV_RECORD_WARMUP_TIMEOUT  seconds to wait for /dv/status  (default 90)
#   DV_RECORD_MIN_FREE_GB     refuse to record below this free space (default 20)
# Opt-out: `touch /etc/dv/norecord` (isc-owned dir) disables recording.
#
# The unit sends SIGINT on stop (KillSignal) so rosbag2 finalizes the bag. A
# hard kill — or copying the bag while it is still recording — leaves an
# unfinalized mcap (header magic present, footer/index missing). Always
# `dv stop` before scp'ing a bag off the car.
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh — `dv update` alone does NOT refresh it).
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
EXCLUDE="${DV_RECORD_EXCLUDE-}"                       # default: exclude NOTHING
BEST_EFFORT="${DV_RECORD_BEST_EFFORT-/lidar_points}"
WARMUP_TIMEOUT="${DV_RECORD_WARMUP_TIMEOUT:-90}"
MIN_FREE_GB="${DV_RECORD_MIN_FREE_GB:-20}"

mkdir -p "$OUTDIR"

free_gb=$(( $(df -Pk "$OUTDIR" | awk 'NR==2{print $4}') / 1024 / 1024 ))
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  LOG "only ${free_gb} GB free in $OUTDIR (< ${MIN_FREE_GB}) — refusing to record."
  exit 1
fi

# Warmup: wait for mission_control's /dv/status heartbeat (the last node up).
# /dv/status is RELIABLE so `topic echo` works on it — unlike the best-effort
# cloud, which a default (reliable) reader would never see.
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

# Per-topic QoS overrides: force a best-effort reader on the sensor topics that
# are published best-effort (see the QoS note above). depth 10 on a 4.5 MB
# cloud is already a ~45 MB buffer — do not raise it casually.
QOS_ARGS=()
QOS_FILE=""
if [ -n "$BEST_EFFORT" ]; then
  QOS_FILE=$(mktemp /tmp/dv_record_qos.XXXXXX.yaml) || QOS_FILE=""
  if [ -n "$QOS_FILE" ]; then
    for t in $BEST_EFFORT; do
      printf '%s:\n  reliability: best_effort\n  history: keep_last\n  depth: 10\n  durability: volatile\n' "$t"
    done > "$QOS_FILE"
    QOS_ARGS=(--qos-profile-overrides-path "$QOS_FILE")
    trap 'rm -f "$QOS_FILE"' EXIT
    LOG "best-effort reader forced on: $BEST_EFFORT"
  fi
fi

MODE=$(cat /run/dv_mode 2>/dev/null || echo run)
OUT="$OUTDIR/${MODE}_$(date +%Y%m%d_%H%M%S)"
if [ -n "$EXCLUDE" ]; then
  LOG "recording -> $OUT (all topics except '$EXCLUDE')  [${free_gb} GB free]"
  exec ros2 bag record -s mcap -o "$OUT" "${QOS_ARGS[@]}" -a -x "$EXCLUDE"
else
  LOG "recording -> $OUT (ALL topics, incl. /lidar_points ~45 MB/s)  [${free_gb} GB free]"
  exec ros2 bag record -s mcap -o "$OUT" "${QOS_ARGS[@]}" -a
fi
