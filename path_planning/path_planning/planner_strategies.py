"""Per-mission path planner strategies (FaSTTUBe MissionTypes)."""
from __future__ import annotations

from fsd_path_planning import MissionTypes

from node_base.base_lifecycle_node import ExecutionStrategy


class PathPlannerStrategy(ExecutionStrategy):
    def get_mission_type(self) -> MissionTypes:
        raise NotImplementedError

    def is_deterministic(self) -> bool:
        """True for missions whose path is fully rule-defined and driven from a
        fixed reference instead of the FaSTTUBe cone planner. Only skidpad, for
        now — the node routes these to the deterministic runtime and never
        enters the FaSTTUBe code path."""
        return False

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

    def is_deterministic(self) -> bool:
        # Skidpad is driven from the fixed FS-Rules figure-eight reference
        # (skidpad package), not FaSTTUBe. See SkidpadPlannerRuntime.
        return True


PATH_PLANNING_STRATEGY_MAP: dict[str, type[PathPlannerStrategy]] = {
    "trackdrive": TrackdrivePathPlanner,
    "autocross": AutocrossPathPlanner,
    "accel": AccelPathPlanner,
    "skidpad": SkidpadPathPlanner,
}
