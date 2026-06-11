"""Per-mission path planner strategies (FaSTTUBe MissionTypes)."""
from __future__ import annotations

from fsd_path_planning import MissionTypes

from node_base.base_lifecycle_node import ExecutionStrategy


class PathPlannerStrategy(ExecutionStrategy):
    def get_mission_type(self) -> MissionTypes:
        raise NotImplementedError

    def execute(self) -> None:
        pass


class TrackdrivePathPlanner(PathPlannerStrategy):
    def get_mission_type(self) -> MissionTypes:
        return MissionTypes.trackdrive


class AutocrossPathPlanner(PathPlannerStrategy):
    def get_mission_type(self) -> MissionTypes:
        return MissionTypes.trackdrive


class AccelPathPlanner(PathPlannerStrategy):
    def get_mission_type(self) -> MissionTypes:
        return MissionTypes.acceleration


class SkidpadPathPlanner(PathPlannerStrategy):
    def get_mission_type(self) -> MissionTypes:
        return MissionTypes.skidpad


class ScrutiPathPlanner(PathPlannerStrategy):
    """Conservative planning — reuse skidpad mission type in FaSTTUBe."""

    def get_mission_type(self) -> MissionTypes:
        return MissionTypes.skidpad


PATH_PLANNING_STRATEGY_MAP: dict[str, type[PathPlannerStrategy]] = {
    "trackdrive": TrackdrivePathPlanner,
    "autocross": AutocrossPathPlanner,
    "accel": AccelPathPlanner,
    "skidpad": SkidpadPathPlanner,
    "scruti": ScrutiPathPlanner,
}
