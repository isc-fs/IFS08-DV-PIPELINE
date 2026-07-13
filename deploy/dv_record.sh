#!/usr/bin/env bash
# =============================================================================
# dv_record.sh — race SENSOR rosbag recorder (run by dv-record.service).
#
# Purpose: capture a *replayable HIL sensor source* on every racing start, so a
# race can later be re-driven through the pipeline off-car (bench replay).
#
# It therefore records ONLY the raw sensor inputs the autonomy consumes —
# by default /lidar_points (Hesai PointCloud2) + /imu (uDV) — and NOTHING the
# pipeline itself produces. That is deliberate:
#   * A bag of the whole graph (/odom, /tf, /ctrl/cmd, /dv/status, …) cannot be
#     replayed into a live pipeline — every recorded output double-publishes on
#     top of the live node, stale /tf breaks the frame tree (Foxglove goes
#     blank), and the pipeline is fed contradictory state so it commands no
#     motion while still reaching DRIVING cleanly. A sensor-only bag replays
#     the way bench_replay.sh expects: `--remap /imu:=/imu_bag
#     /lidar_points:=/lidar_points_bag` + the re-stamp relay.
#   * /lidar_packets_loss is a driver diagnostic, NOT lidar data — a whitelist
#     never captures it, so it can't masquerade as the cloud in Foxglove.
#
# QoS: the Hesai cloud is published BEST_EFFORT (SensorDataQoS). `ros2 bag
# record` defaults to a RELIABLE subscription, which is QoS-incompatible and
# would record ZERO cloud frames. We therefore force each recorded topic to a
# best-effort reader via --qos-profile-overrides-path (a best-effort reader is
# compatible with both best-effort and reliable writers, so this is safe for
# every sensor topic).
#
# Started automatically whenever dv-pipeline.service starts (race boot,
# `dv race`, `dv restart`); PartOf= stops it with the pipeline; KillSignal=
# SIGINT so rosbag2 finalizes metadata.yaml (a hard kill corrupts the bag —
# recover with `rm metadata.yaml && ros2 bag reindex <bag> -s mcap`).
#
# Knobs (systemd drop-in or environment):
#   DV_RECORD_DIR             output dir              (default /home/isc/bags)
#   DV_RECORD_TOPICS          space-separated whitelist
#                             (default "/lidar_points /imu")
#   DV_RECORD_WARMUP_TIMEOUT  seconds to wait for /dv/status  (default 90)
#   DV_RECORD_MIN_FREE_GB     refuse to record below this free space
#                             (default 20 — the cloud is ~45 MB/s ≈ 2.7 GB/min)
# Opt-out: `touch /etc/dv/norecord`.
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
TOPICS="${DV_RECORD_TOPICS:-/lidar_points /imu}"
WARMUP_TIMEOUT="${DV_RECORD_WARMUP_TIMEOUT:-90}"
MIN_FREE_GB="${DV_RECORD_MIN_FREE_GB:-20}"

mkdir -p "$OUTDIR"

free_gb=$(( $(df -Pk "$OUTDIR" | awk 'NR==2{print $4}') / 1024 / 1024 ))
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  LOG "only ${free_gb} GB free in $OUTDIR (< ${MIN_FREE_GB}) — refusing to record."
  exit 1
fi

# Warmup: wait for mission_control's /dv/status heartbeat. It is RELIABLE (so
# `topic echo` works — unlike the best-effort cloud, which would hang a default
# reader) and it is the last node up, so its appearance means the sensor graph
# is live.
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
sleep 2   # let the sensor publishers register with discovery

# Best-effort QoS override for every recorded topic (mandatory for the
# best-effort Hesai cloud; harmless for reliable publishers like /imu).
QOS=$(mktemp /tmp/dv_record_qos.XXXXXX.yaml)
{
  for t in $TOPICS; do
    printf '%s:\n  reliability: best_effort\n  history: keep_last\n  depth: 100\n  durability: volatile\n' "$t"
  done
} > "$QOS"
trap 'rm -f "$QOS"' EXIT

MODE=$(cat /run/dv_mode 2>/dev/null || echo run)
OUT="$OUTDIR/${MODE}_$(date +%Y%m%d_%H%M%S)"
LOG "recording sensor bag -> $OUT   topics: $TOPICS"
exec ros2 bag record -s mcap -o "$OUT" \
     --qos-profile-overrides-path "$QOS" \
     $TOPICS
