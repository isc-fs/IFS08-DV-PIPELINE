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

# Cone height encodes type per Cone_Detection (cone_detection_node.py:140).
# Big-orange cones (start-finish) are ≥ 0.30 m; small cones ~0.20 m.
BIG_ORANGE_HEIGHT_THRESHOLD_M = 0.30

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
