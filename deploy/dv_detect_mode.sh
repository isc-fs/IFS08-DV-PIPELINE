#!/usr/bin/env bash
# =============================================================================
# dv_detect_mode.sh — print the DVPC boot mode: "umbilical" | "race".
#
# Precedence:
#   1. Explicit override file (DV_MODE_OVERRIDE_FILE, default /etc/dv/mode):
#        "race" / "umbilical"  → force that mode, ignore the cable.
#        "auto" / empty / absent / unknown → fall through to autodetect.
#      Set it with:  dv mode {race|umbilical|auto}
#   2. enp2s0 carrier autodetect:
#        "umbilical" — cable present on enp2s0 (bench: config / debug / monitor)
#        "race"      — nothing on enp2s0 (on-car: normal racing operation)
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh). Kept side-effect-free: it only prints.
# =============================================================================
IFACE="${DV_UMBILICAL_IFACE:-enp2s0}"
OVERRIDE_FILE="${DV_MODE_OVERRIDE_FILE:-/etc/dv/mode}"

if [ -r "$OVERRIDE_FILE" ]; then
  ov=$(tr '[:upper:]' '[:lower:]' < "$OVERRIDE_FILE" 2>/dev/null | tr -d '[:space:]')
  case "$ov" in
    race|umbilical) echo "$ov"; exit 0 ;;
    *)              : ;;  # auto/empty/unknown → autodetect below
  esac
fi

carrier=$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo 0)
[ "$carrier" = "1" ] && echo umbilical || echo race
