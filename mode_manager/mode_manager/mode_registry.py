"""
Mode registry — single source of truth for mode → node → behavior mapping.

mode_manager calls ~/setup on each node with (mode_name, behavior) before
configure. Nodes inherit BaseLifecycleNode and pick strategies from
`behavior` in on_configure (odometry_filter_node uses behavior ``base``
as a reserved no-op today).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class NodeModeConfig:
    node_name: str
    behavior: str


@dataclass(frozen=True)
class ModeDefinition:
    mode_name: str
    mission_id: int
    nodes: tuple[NodeModeConfig, ...]


AUTONOMY_NODE_ORDER: tuple[str, ...] = (
    "odometry_filter_node",
    "cone_detection_node",
    "slam_node",
    "path_planning_node",
    "control_node",
)


def _node(node_name: str, behavior: str) -> NodeModeConfig:
    return NodeModeConfig(node_name=node_name, behavior=behavior)


MODE_REGISTRY: Mapping[str, ModeDefinition] = MappingProxyType({
    "trackdrive": ModeDefinition(
        mode_name="trackdrive",
        mission_id=1,
        nodes=(
            _node("odometry_filter_node", "base"),
            _node("cone_detection_node", "base"),
            _node("slam_node", "trackdrive"),
            _node("path_planning_node", "trackdrive"),
            _node("control_node", "pure_pursuit"),
        ),
    ),
    "autocross": ModeDefinition(
        mode_name="autocross",
        mission_id=2,
        nodes=(
            _node("odometry_filter_node", "base"),
            _node("cone_detection_node", "base"),
            _node("slam_node", "autocross"),
            _node("path_planning_node", "autocross"),
            # Switched stanley → pure_pursuit 2026-05-26 after bag
            # _221138 analysis. Stanley's heading-error term ψ_e enters
            # the formula in radians with NO gain, and during transient
            # FS-DV cornering ψ_e easily reaches 30-40° on its own —
            # saturating the actuator before the cross-track term even
            # plays. Tuning stanley_k (cross-track gain) didn't help
            # (the dominant term wasn't gain-tunable); enabling
            # stanley_k_yaw_rate produced under-correction in normal
            # cornering. Pure Pursuit has the β·R radius cap
            # (lookahead_radius_factor=0.7) which is the canonical
            # geometric guard against the late-corner hairpin
            # oversteer pattern. Trackdrive + accel already use PP for
            # this reason. Skidpad keeps Stanley because its constant-
            # curvature figure-8 is where Stanley's nearest-projection
            # geometry shines (no chase-target tuning).
            _node("control_node", "pure_pursuit"),
        ),
    ),
    "accel": ModeDefinition(
        mode_name="accel",
        mission_id=3,
        nodes=(
            _node("odometry_filter_node", "base"),
            _node("cone_detection_node", "base"),
            _node("slam_node", "accel"),
            _node("path_planning_node", "accel"),
            _node("control_node", "pure_pursuit"),
        ),
    ),
    "skidpad": ModeDefinition(
        mode_name="skidpad",
        mission_id=4,
        nodes=(
            _node("odometry_filter_node", "base"),
            _node("cone_detection_node", "base"),
            _node("slam_node", "skidpad"),
            _node("path_planning_node", "skidpad"),
            _node("control_node", "stanley"),
        ),
    ),
    "scruti": ModeDefinition(
        mode_name="scruti",
        mission_id=5,
        nodes=(
            _node("odometry_filter_node", "base"),
            _node("cone_detection_node", "base"),
            _node("slam_node", "scruti"),
            _node("path_planning_node", "scruti"),
            _node("control_node", "stanley"),
        ),
    ),
})


MISSION_ID_TO_NAME: Mapping[int, str] = MappingProxyType({
    m.mission_id: m.mode_name for m in MODE_REGISTRY.values()
})

MISSION_NAME_TO_ID: Mapping[str, int] = MappingProxyType({
    m.mode_name: m.mission_id for m in MODE_REGISTRY.values()
})


def node_config_for(mode_name: str, node_name: str) -> NodeModeConfig:
    mode = MODE_REGISTRY[mode_name]
    for cfg in mode.nodes:
        if cfg.node_name == node_name:
            return cfg
    return NodeModeConfig(node_name=node_name, behavior=mode_name)
