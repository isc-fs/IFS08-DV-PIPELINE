#!/usr/bin/env bash
# =============================================================================
# install_ptp4l.sh — install the gPTP master config + service on the DVPC.
#
# The Hesai ATX lidar runs 802.1AS/gPTP; this makes ptp4l speak the matching
# profile so the lidar slaves to the DVPC system clock (CLOCK_REALTIME), the
# same clock the micro-ROS IMU (uDV) uses. See docs/lidar_imu_time_sync_mitigation.md.
#
# Installs (with timestamped backups into ~isc/.isc_backups/<ts>/):
#   deploy/ptp4l.conf    → /etc/linuxptp/ptp4l.conf
#   deploy/ptp4l.service → /etc/systemd/system/ptp4l.service
# then enables + (re)starts the service.
#
# Run ON the DVPC, as root, from the updated checkout:
#   sudo /home/isc/dv_ws/src/IFS08-DV-PIPELINE/deploy/install_ptp4l.sh
# Rollback: restore ptp4l.conf from the printed backup dir and `systemctl restart ptp4l`.
# =============================================================================
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)

TS=$(date +%Y%m%d_%H%M%S)
BK="/home/isc/.isc_backups/$TS"
mkdir -p "$BK"

for f in /etc/linuxptp/ptp4l.conf /etc/systemd/system/ptp4l.service; do
  [ -f "$f" ] && cp -a "$f" "$BK/" && echo "backed up $f -> $BK/"
done

install -D -m 644 "$HERE/ptp4l.conf"    /etc/linuxptp/ptp4l.conf
install -m 644     "$HERE/ptp4l.service" /etc/systemd/system/ptp4l.service

systemctl daemon-reload
systemctl enable ptp4l
systemctl restart ptp4l
sleep 2
systemctl is-active --quiet ptp4l && echo "ptp4l active (gPTP master)." \
  || { echo "ptp4l failed to start — check: journalctl -u ptp4l -n 30"; exit 2; }

echo
echo "Done. Verify the lidar is slaving with:"
echo "  sudo pmc -u -b 0 -f /etc/linuxptp/ptp4l.conf 'GET PORT_DATA_SET'   # delayMechanism 2 (P2P), MASTER"
echo "  # then query the ATX ptp_status via PTC (GetLidarStatus byte 52 -> 1=Tracking)"
