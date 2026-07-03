#!/usr/bin/env bash
# =============================================================================
# dv_mode_boot.sh — DVPC boot-mode controller (run once at boot by dv-mode.service)
#
#   RACE MODE (no umbilical on enp2s0):
#     → start the racing operation (dv-pipeline: sensors + real autonomy stack)
#       immediately. No update, no prompt.
#
#   UMBILICAL MODE (cable on enp2s0):
#     → auto-update the pipeline (config/debug/monitor context), then STOP.
#       Racing is NOT started here — the login prompt (or `dv race`) starts it
#       only if the user chooses to race.
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh).
# =============================================================================
set -uo pipefail
LOG(){ logger -t dv-mode "$*" 2>/dev/null; echo "[dv-mode] $*"; }

IFACE="${DV_UMBILICAL_IFACE:-enp2s0}"
CARRIER_WAIT="${DV_CARRIER_WAIT:-12}"

# enp2s0 is 'optional' (so race boot doesn't stall), which means network-online
# does NOT wait for it — and on a fresh boot the PHY carrier can lag a few seconds
# behind interface-up. Without a settle window we'd mis-read a connected bench as
# 'race'. Wait up to CARRIER_WAIT seconds for the umbilical link, breaking as soon
# as it appears (race/on-car: no cable → full wait, then proceed to race).
LOG "Waiting up to ${CARRIER_WAIT}s for $IFACE link to settle…"
for _ in $(seq 1 "$CARRIER_WAIT"); do
  [ "$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo 0)" = "1" ] && break
  sleep 1
done

MODE=$(DV_UMBILICAL_IFACE="$IFACE" /usr/local/bin/dv_detect_mode.sh)
echo "$MODE" > /run/dv_mode
chmod 644 /run/dv_mode 2>/dev/null || true
LOG "Detected mode: $MODE"

if [ "$MODE" = "race" ]; then
  LOG "RACE MODE — starting racing operation (dv-pipeline)."
  systemctl start dv-pipeline.service
  exit 0
fi

# ---- UMBILICAL MODE --------------------------------------------------------
# enp2s0 is 'optional' so boot doesn't stall; give the link/internet a short
# window to come up before the auto-update (umbilical implies a cable is there).
LOG "UMBILICAL MODE — waiting briefly for internet, then auto-updating pipeline."
for _ in $(seq 1 15); do
  timeout 4 getent hosts github.com >/dev/null 2>&1 && break
  sleep 2
done
systemctl start dv-pipeline-update.service || LOG "auto-update trigger failed (non-fatal)."
LOG "UMBILICAL MODE ready — racing held for user choice (login prompt / 'dv race')."
exit 0
