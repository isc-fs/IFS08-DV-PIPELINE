"""Cone color classification — spatial Y-sign + big-orange height.

Replicates the rule used by Publicar_Mapa in pipeline/slam/slam/slam.py
(see _lookup_or_classify) so cone-graph SLAM doesn't depend on the
existing Publicar_Mapa node. Cones are classified at observation time
based on:
  - height (from /Conos_raw marker.scale.z, encoded by Cone_Detection):
      ≥ BIG_ORANGE_HEIGHT_THRESHOLD_M  → big-orange (start/finish)
  - else, sign of Y in body frame (REP-103: +Y is left, -Y is right):
      y > 0 → blue (left side of track)
      y < 0 → yellow (right side of track)
      y ≈ 0 → orange (small orange — center cones)

The "ignore the |y| < ε band" rule keeps near-centerline observations
from flipping color frame-to-frame as the car drifts laterally; for FS
courses this is a fine first cut. The original Publicar_Mapa kept a
world-position color cache (snap radius 0.5 m) on top of this rule;
once the SLAM has persistent landmark IDs, the per-landmark color is
locked at the FIRST observation and never re-classified — that gives
the same effect more cleanly.
"""

from __future__ import annotations

from enum import IntEnum

# Cone height encodes type per Cone_Detection (cone_detection_node.py:140):
# the cluster-height of the LiDAR returns from each cone.
#
# Per FS Driverless cone spec (DS Table 1):
#   - Small cones (yellow / blue / orange):  325 mm
#   - Big-orange (start / finish):           505 mm
#
# The previous threshold of 0.30 m sat *below* the small-cone height of
# 0.325 m, so any LiDAR measurement noise on a small cone tipped it
# above 0.30 m and got it classified BIG_ORANGE. On the live UE5 run on
# 2026-04-29 the very first cones the car drove past read back as
# 0.32 / 0.34 m, control's `_compute_orange_stop_distance` saw them as
# a finish gate at 30 m forward, latched a stop target, fired EBS, and
# bridged-engaged the handbrake — all on harmless small cones.
#
# 0.40 m sits halfway between 0.325 and 0.505 m and gives ~7.5 cm of
# noise budget on either side. If small cones ever measure that tall
# we have a measurement problem upstream of here.
BIG_ORANGE_HEIGHT_THRESHOLD_M = 0.40

# Lateral band where small cones get orange (centerline) instead of
# blue/yellow. Tuned to match typical FS lane width — anything within
# ±0.5 m of the car's longitudinal axis is orange-class.
ORANGE_CENTERLINE_BAND_M = 0.5


class ConeColor(IntEnum):
    """Cone color classes used as data-association gates.

    Stored as int so they fit in compact dicts/arrays and pickle.
    """

    YELLOW = 0       # right-side of track
    BLUE = 1         # left-side of track
    ORANGE = 2       # small orange — centerline / waypoint markers
    BIG_ORANGE = 3   # large orange — start/finish line


def classify(body_y: float, height: float) -> ConeColor:
    """Classify a single cone observation in base_link frame.

    Args:
        body_y: Y position of the cone in base_link (REP-103: +Y left).
        height: cone height encoded by Cone_Detection on marker.scale.z.

    Returns:
        ConeColor enum value.
    """
    if height >= BIG_ORANGE_HEIGHT_THRESHOLD_M:
        return ConeColor.BIG_ORANGE
    if body_y > ORANGE_CENTERLINE_BAND_M:
        return ConeColor.BLUE
    if body_y < -ORANGE_CENTERLINE_BAND_M:
        return ConeColor.YELLOW
    return ConeColor.ORANGE
