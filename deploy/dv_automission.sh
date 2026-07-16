#!/usr/bin/env bash
# =============================================================================
# dv_automission.sh — auto-record manual driving when Manual is confirmed on AMI.
#
# Always-on watcher (dv-automission.service, enabled at boot). Subscribes to the
# uDV's /ami/mission and reconciles the manual recorder to it:
#
#   mission 0 (Manual CONFIRMED) + pipeline NOT running → start dv-manual.service
#   mission 1..9 (a real DV mission confirmed)          → stop  dv-manual.service
#   mission -1 (up, nothing confirmed yet)              → leave state untouched
#   /ami/mission not seen (uDV/agent silent)            → leave state untouched
#   /etc/dv/norecord present                            → stop + stay idle
#
# ONE OPERATOR STEP (uDV#189): the AMI board only puts an index on the bus after
# the operator CONFIRMS a mission — it does NOT broadcast the dial position. So
# /ami/mission latches -1 until a mission is confirmed; confirming Manual makes
# it a latched 0, republished at ~10 Hz with the ASMS off. There is therefore no
# true "power-on-at-Manual = record" zero-touch: reaching a positive 0 costs one
# button press (select Manual + confirm). True zero-touch would need the AMI
# board to broadcast the dial index continuously — an IFS08-DV_AMI change, not a
# uDV one. -1 is deliberately NOT treated as Manual (it's also the pre-arm state
# of every mission and the fresh-boot state).
#
# After confirming Manual → records, with NO pipeline and NO commands to the car.
# Confirm a real mission + `dv race` → this releases the recorder (pipeline's own
# recorder takes over); confirm Manual again after `dv stop` → it resumes.
#
# It never sends anything to the car; it only reads /ami/mission and starts/stops
# a local systemd unit.
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
    ""|-*)
      # "" = uDV/agent silent; -1 = uDV+agent UP but NO mission confirmed on the
      # AMI yet (fresh boot, or post-reset). Per uDV#189: /ami/mission latches
      # -1 until the operator *confirms* a mission on the AMI — Manual is index
      # 0 only AFTER it is confirmed, it is NOT emitted from the dial position.
      # So -1 is NOT "Manual" and must not start OR stop recording: leave the
      # current state as-is (don't force-drop an in-progress manual bag on a
      # transient reset, don't start one on a bare boot).
      : ;;
    0)
      # Manual was CONFIRMED on the AMI (dial to Manual + press confirm) — the
      # uDV now latches 0 and republishes it at ~10 Hz, ASMS-off. Record unless
      # the pipeline owns the sensors.
      if pipeline_active; then
        : # a real run is up; the pipeline's recorder is in charge
      elif ! manual_active; then
        LOG "AMI Manual (0) confirmed, no pipeline → starting manual recording"
        sudo -n systemctl start dv-manual.service || LOG "failed to start dv-manual.service"
      fi ;;
    *)
      # A real DV mission (1..9) was CONFIRMED → the manual recorder must
      # release (whoever runs `dv race` gets a clean Hesai driver + its own
      # recorder). Note: -1 is handled above, so this is only positive indices.
      if manual_active; then
        LOG "AMI mission $m confirmed → stopping manual recording"
        sudo -n systemctl stop dv-manual.service || LOG "failed to stop dv-manual.service"
      fi ;;
  esac

  sleep "$POLL"
done
