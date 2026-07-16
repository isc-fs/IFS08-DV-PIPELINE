#!/usr/bin/env bash
# =============================================================================
# dv_automission.sh — auto-record manual driving when the AMI selects Manual.
#
# Always-on watcher (dv-automission.service, enabled at boot). Subscribes to the
# uDV's /ami/mission and reconciles the manual recorder to it:
#
#   mission 0 (Manual) + pipeline NOT running  →  start dv-manual.service
#   mission 1..9 (a real DV mission)           →  stop  dv-manual.service
#   /ami/mission not seen (uDV/agent silent)   →  leave current state untouched
#   /etc/dv/norecord present                   →  stop + stay idle (opt-out)
#
# So: power on with the dial at Manual (index 0 = the default) → the manual
# sensor bag records itself, with NO pipeline and NO commands to the car. Select
# a real mission and `dv race` → this releases the recorder (pipeline's own
# recorder takes over); dial back to Manual after `dv stop` → it resumes.
#
# ⚠ BLOCKED ON uDV: this only fires if the uDV publishes /ami/mission with the
# ASMS OFF and no pipeline (isc-fs/IFS08-DV-uDV#189). Until then /ami/mission is
# simply never seen in this state and the watcher stays idle — inert, not
# broken. It never sends anything to the car; it only reads /ami/mission and
# starts/stops a local systemd unit.
#
# Reconcile loop (not edge-triggered): re-evaluates every DV_AUTOMISSION_POLL
# seconds so the pipeline-guard is rechecked too — e.g. dial-to-Manual while the
# pipeline is still up must not start the recorder, but a later `dv stop` must.
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

manual_active()  { systemctl is-active --quiet dv-manual.service; }
pipeline_active(){ systemctl is-active --quiet dv-pipeline.service; }

# Latest /ami/mission value as a bare integer, or empty if not seen. /ami/mission
# is BEST_EFFORT (UPLINK_QOS) — a default reliable reader sees NOTHING — so force
# a best-effort reader. Parse `data: N` (robust across ros2 CLI versions).
read_mission(){
  timeout 3 ros2 topic echo --once --qos-reliability best_effort /ami/mission \
    2>/dev/null | awk '/^data:/{gsub(/[^0-9-]/,"",$2); print $2; exit}'
}

LOG "watching /ami/mission (poll ${POLL}s); Manual(0)+no-pipeline → record. Blocked on uDV#189 until /ami/mission publishes with ASMS off."

while true; do
  if [ -f /etc/dv/norecord ]; then
    manual_active && { LOG "/etc/dv/norecord set → stopping manual recorder"; sudo -n systemctl stop dv-manual.service; }
    sleep "$POLL"; continue
  fi

  m=$(read_mission)
  case "$m" in
    "")
      # uDV/agent silent — we don't know the mission. Do NOT thrash the
      # recorder on a transient blip; leave whatever is running as-is.
      : ;;
    0)
      # Manual selected. Record unless the pipeline owns the sensors.
      if pipeline_active; then
        : # a real run is up; the pipeline's recorder is in charge
      elif ! manual_active; then
        LOG "AMI Manual (0) selected, no pipeline → starting manual recording"
        sudo -n systemctl start dv-manual.service || LOG "failed to start dv-manual.service"
      fi ;;
    *)
      # A real DV mission is dialled in → the manual recorder must release
      # (whoever runs `dv race` gets a clean Hesai driver + its own recorder).
      if manual_active; then
        LOG "AMI mission $m selected → stopping manual recording"
        sudo -n systemctl stop dv-manual.service || LOG "failed to stop dv-manual.service"
      fi ;;
  esac

  sleep "$POLL"
done
