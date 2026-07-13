#!/usr/bin/env bash
# =============================================================================
# install_dv_pipeline_service.sh — switch DVPC race mode to the real pipeline.
#
# Installs (with timestamped backups into ~isc/.isc_backups/<ts>/):
#   deploy/dv-pipeline.service → /etc/systemd/system/   (NOT enabled — started
#                                by dv-mode at race boot, or manually: dv race)
#   deploy/dv-record.service   → /etc/systemd/system/   (enabled: WantedBy=
#                                dv-pipeline.service → records a sensor bag on
#                                every racing start, after pipeline warmup)
#   deploy/dv_record.sh        → /usr/local/bin/        (warmup wait + record
#                                /lidar_points+/imu, best-effort QoS)
#   deploy/dv                  → /usr/local/bin/dv      (targets dv-pipeline)
#   deploy/dv_mode_boot.sh     → /usr/local/bin/        (race → dv-pipeline, verified)
#   deploy/dv_detect_mode.sh   → /usr/local/bin/        (mode detect + override)
#   deploy/dv-race.sudoers     → /etc/sudoers.d/dv-race (NOPASSWD dv-pipeline)
# creates the isc-owned override dir /etc/dv (so `dv mode …` needs no sudo),
# and re-points the login prompt (/etc/profile.d/zz-dv-mode-prompt.sh) from
# isc-startup to dv-pipeline. The legacy isc-startup.service (isc_ws stub
# stack) is left installed but nothing references it anymore.
#
# Run ON the DVPC, as root, from the updated checkout:
#   sudo /home/isc/dv_ws/src/IFS08-DV-PIPELINE/deploy/install_dv_pipeline_service.sh
# =============================================================================
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)

TS=$(date +%Y%m%d_%H%M%S)
BK="/home/isc/.isc_backups/$TS"
mkdir -p "$BK"

for f in /etc/systemd/system/dv-pipeline.service \
         /etc/systemd/system/dv-record.service \
         /usr/local/bin/dv \
         /usr/local/bin/dv_mode_boot.sh \
         /usr/local/bin/dv_detect_mode.sh \
         /usr/local/bin/dv_record.sh \
         /etc/profile.d/zz-dv-mode-prompt.sh; do
  [ -f "$f" ] && cp -a "$f" "$BK/" && echo "backed up $f"
done

install -m 644 "$HERE/dv-pipeline.service" /etc/systemd/system/dv-pipeline.service
install -m 644 "$HERE/dv-record.service"   /etc/systemd/system/dv-record.service
install -m 755 "$HERE/dv"                  /usr/local/bin/dv
install -m 755 "$HERE/dv_mode_boot.sh"     /usr/local/bin/dv_mode_boot.sh
install -m 755 "$HERE/dv_detect_mode.sh"   /usr/local/bin/dv_detect_mode.sh
install -m 755 "$HERE/dv_record.sh"        /usr/local/bin/dv_record.sh

# Boot-mode override dir: isc-owned so `dv mode {race|umbilical|auto}` can
# write /etc/dv/mode without sudo. dv_detect_mode.sh reads it (root) at boot.
install -d -m 755 -o isc -g isc /etc/dv

# Passwordless start/stop/restart of the racing unit for the [r] login prompt
# and `dv race`. This NOPASSWD rule MUST name dv-pipeline.service — the earlier
# isc-startup→dv-pipeline switch left it pointing at the old unit, which made
# `sudo -n systemctl start dv-pipeline.service` prompt for a password and both
# `[r]` and `dv race` fail with "Could not start automatically". Validate with
# visudo before moving into place so a bad rule can never lock out sudo.
[ -f /etc/sudoers.d/dv-race ] && cp -a /etc/sudoers.d/dv-race "$BK/"
visudo -cf "$HERE/dv-race.sudoers"
install -m 0440 -o root -g root "$HERE/dv-race.sudoers" /etc/sudoers.d/dv-race
visudo -c >/dev/null

# Login prompt: the [r] RACE choice + status line must start the real pipeline.
if [ -f /etc/profile.d/zz-dv-mode-prompt.sh ]; then
  sed -i 's/isc-startup/dv-pipeline/g' /etc/profile.d/zz-dv-mode-prompt.sh
  echo "re-pointed zz-dv-mode-prompt.sh at dv-pipeline"
fi

systemctl daemon-reload

# Recorder rides along with the pipeline: enable creates the
# dv-pipeline.service.wants/ symlink (WantedBy=dv-pipeline.service), so every
# racing start also starts dv-record. dv-record itself stays un-started here —
# updating ≠ running, same policy as dv-pipeline.
systemctl enable dv-record.service >/dev/null 2>&1 || systemctl enable dv-record.service

echo
echo "Done. Race mode / 'dv race' now starts dv-pipeline.service"
echo "(bringup car_bringup.launch.py: Hesai + TFs + real autonomy)."
echo "Rollback: restore from $BK"
