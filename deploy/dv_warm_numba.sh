#!/usr/bin/env bash
# =============================================================================
# dv_warm_numba.sh — pre-bake the Numba JIT cache for the autonomy nodes.
#
# The pipeline's Numba kernels (cone_detection's RANSAC/GN-fit and
# fsd_path_planning's ~92 sorting/matching/parameterization kernels) compile
# with cache=True into NUMBA_CACHE_DIR (pinned in dv-pipeline.service — keep
# the default below in sync). The cache is keyed on source content/mtime, so
# it is invalidated exactly when `dv update` / the umbilical auto-update
# rewrites sources. Running this right after an update — on the bench, with
# the umbilical connected, nobody waiting on the car — means race-mode
# bring-up always hits the warm path (~1 s per node) instead of paying the
# cold compile (~1-2 min, measured 63 s for fsd_path_planning alone) inside
# mode_manager's 120 s per-node configure budget.
#
# Called from:
#   - dv-pipeline-update.service drop-in (ExecStartPost, installed by
#     install_dv_pipeline_service.sh) — after every umbilical auto-update
#   - `dv update` — after every manual update
#   - install_dv_pipeline_service.sh — once at install
#
# Best-effort by design: ALWAYS exits 0. A failed warmup only means the
# first mission bring-up pays the JIT during Phase-1 warming_up, exactly as
# it would without this script.
# =============================================================================
set -uo pipefail
LOG(){ logger -t dv-warm-numba "$*" 2>/dev/null; echo "[warm-numba] $*"; }

# Keep in sync with Environment=NUMBA_CACHE_DIR in dv-pipeline.service —
# baking into a different dir than the nodes read would warm nothing.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/home/isc/.cache/dv_numba}"

# The cache must be owned by the user the pipeline runs as (User=isc in
# dv-pipeline.service). Root-owned cache files are readable but block the
# nodes' own cache WRITES — numba fails silently and re-JITs — so when the
# installer or a root service calls us, re-exec as isc.
if [ "$(id -u)" = 0 ]; then
  exec runuser -u isc -- /usr/bin/env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" bash "$0" "$@"
fi

source /opt/ros/humble/setup.bash 2>/dev/null || { LOG "no ROS humble — skipping"; exit 0; }
# ros2_ws (hesai driver) is optional for the warmup; dv_ws is not.
source /home/isc/ros2_ws/install/local_setup.bash 2>/dev/null || true
source /home/isc/dv_ws/install/local_setup.bash 2>/dev/null \
  || { LOG "no dv_ws install — skipping"; exit 0; }

LOG "baking Numba cache into $NUMBA_CACHE_DIR (fast if already warm)…"
python3 - <<'EOF' || LOG "warmup failed (non-fatal — first bring-up will JIT instead)"
import time

t0 = time.perf_counter()
from cone_detection.cone_detection import warmup_numba_functions
warmup_numba_functions()
print(f"[warm-numba] cone_detection: {time.perf_counter() - t0:.1f}s", flush=True)

t0 = time.perf_counter()
from path_planning.fasttube_adapter import warmup_numba_planner
# Trackdrive is the racing mission; its kernels are the superset shared by
# the sorter/matcher/parameterization stages the other missions also touch.
warmup_numba_planner()
print(f"[warm-numba] fsd_path_planning: {time.perf_counter() - t0:.1f}s", flush=True)
EOF
LOG "done."
exit 0
