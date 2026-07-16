#!/usr/bin/env bash
# =============================================================================
# dv_automission.sh — auto-record manual driving whenever the car is powered.
#
# Always-on watcher (dv-automission.service, enabled at boot). Triggers on the
# uDV SENSOR STREAM being live — i.e. the car is powered and the uDV micro-ROS
# node is up — NOT on any AMI/mission selection. Zero operator steps, no AMI:
#
#   uDV alive (/imu publishing) + pipeline NOT running  →  start dv-manual.service
#   pipeline running (`dv race`)                        →  stop  (pipeline records)
#   uDV silent for a sustained window (car off / down)  →  stop  (finalize the bag)
#   brief uDV blip                                      →  leave state untouched
#   /etc/dv/norecord present                            →  stop + stay idle
#
# WHY car-power, not AMI (uDV#189): the AMI board only puts a mission index on
# the bus after the operator CONFIRMS one, so /ami/mission needs a button press
# to reach "Manual (0)" — not what we want. The uDV's /imu (and /motor_rpm,
# /steering_angle) publish continuously, ASMS-off, with no confirmation and no
# pipeline — so "is the car on?" is the honest zero-touch trigger. Power the car
# → it records; `dv race` → the pipeline's recorder takes over; `dv stop` while
# still powered → manual recording resumes.
#
# It never sends anything to the car; it only reads /imu and starts/stops a
# local systemd unit.
#
# ⚠ DISK: this records from power-on, INCLUDING idle setup time, and the bag
# includes the ~45 MB/s lidar cloud. Long powered-but-idle periods burn disk
# fast — `touch /etc/dv/norecord` to pause, `dv stop` when done, or set
# DV_RECORD_EXCLUDE='^/lidar_points$' for a light telemetry-only bag. The
# DV_RECORD_MIN_FREE_GB floor still refuses to start below the free-space limit.
#
# Reconcile loop (not edge-triggered): re-evaluates every DV_AUTOMISSION_POLL
# seconds so the pipeline-guard is rechecked too — car-on while the pipeline is
# up must not start the recorder, but a later `dv stop` must.
#
# NOTE: the unit/script keep the legacy "automission" name; the trigger is now
# car-power, not mission. A rename to dv-autorecord would be clearer (follow-up).
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh).
# =============================================================================
set -o pipefail
LOG(){ logger -t dv-automission "$*" 2>/dev/null; echo "[dv-automission] $*"; }

source /opt/ros/humble/setup.bash 2>/dev/null
source /home/isc/ros2_ws/install/local_setup.bash 2>/dev/null
source /home/isc/dv_ws/install/local_setup.bash 2>/dev/null
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

POLL="${DV_AUTOMISSION_POLL:-2}"
# Consecutive silent polls before we treat the car as OFF and stop (finalize the
# bag). POLL=2 × 3 = ~6 s: rides out an agent blip, but ends the bag on a real
# power-down. A single blip does NOT stop recording.
SILENCE_LIMIT="${DV_AUTOMISSION_SILENCE_LIMIT:-3}"
# Which uDV topic proves "the car is on". /imu is 400 Hz best-effort — fastest to
# confirm. A default (reliable) reader sees NOTHING from a best-effort publisher.
ALIVE_TOPIC="${DV_AUTOMISSION_ALIVE_TOPIC:-/imu}"

manual_active()  { systemctl is-active --quiet dv-manual.service; }
pipeline_active(){ systemctl is-active --quiet dv-pipeline.service; }

# True if a message actually arrives on the alive-topic within the timeout — i.e.
# the uDV is actively publishing (car powered + agent up). Presence in
# `topic list` is not enough; we want a real sample.
udv_alive(){
  timeout 3 ros2 topic echo --once --qos-reliability best_effort "$ALIVE_TOPIC" \
    >/dev/null 2>&1
}

start_manual(){ sudo -n systemctl start dv-manual.service || LOG "failed to start dv-manual.service"; }
stop_manual(){  sudo -n systemctl stop  dv-manual.service || LOG "failed to stop dv-manual.service"; }

LOG "watching ${ALIVE_TOPIC} (poll ${POLL}s); car-powered + no-pipeline → record. No AMI needed."

silent=0
while true; do
  if [ -f /etc/dv/norecord ]; then
    manual_active && { LOG "/etc/dv/norecord set → stopping manual recorder"; stop_manual; }
    silent=0; sleep "$POLL"; continue
  fi

  if pipeline_active; then
    # A real run owns the sensors; its own recorder is in charge.
    manual_active && { LOG "pipeline up → releasing manual recorder"; stop_manual; }
    silent=0; sleep "$POLL"; continue
  fi

  if udv_alive; then
    silent=0
    if ! manual_active; then
      LOG "car powered (${ALIVE_TOPIC} live), no pipeline → starting manual recording"
      start_manual
    fi
  else
    silent=$((silent + 1))
    if [ "$silent" -ge "$SILENCE_LIMIT" ] && manual_active; then
      LOG "${ALIVE_TOPIC} silent ${silent}×${POLL}s (car off / uDV down) → stopping manual recording"
      stop_manual
    fi
    # Under the limit: a brief blip — leave the recorder as-is.
  fi

  sleep "$POLL"
done
