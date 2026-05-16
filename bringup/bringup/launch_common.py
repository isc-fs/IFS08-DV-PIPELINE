"""Shared launch-file helpers and IFSSIM topic remap constants.

Every launch file under `bringup/launch/` composes the same node-graph
fragments — management trio, autonomy lifecycle nodes, /fsds/* remap
table. Keeping the shared pieces here avoids three near-identical
copies drifting out of sync.

Public API used by the launch files:

  REMAP_*                IFSSIM /fsds/* surface → pipeline-side names.
                         Single source of truth.

  auto_active(...)       Build a LifecycleNode that auto-drives itself
                         from `unconfigured` to `active` at launch
                         start. Used for the management trio whose
                         action/service endpoints must be live before
                         mode_manager fans change_state out to the
                         autonomy nodes.

  autonomy_lifecycle(...) Build a LifecycleNode that parks in
                          `unconfigured` until mode_manager drives the
                          configure/activate transitions in response
                          to a StartMission/activate_mode call.

  autonomy_actions()     The pre-baked list of autonomy
                         lifecycle-nodes (odometry_filter,
                         cone_detection, slam, path_planning,
                         control) with their remappings. Bring-up
                         order is owned by mode_manager's registry.
                         Lets each launch file insert them with one
                         `actions += ...`.

  management_actions(include_sim_supervisor=True)
                         The management trio pre-baked with the right
                         remaps. The sim_supervisor flag toggles
                         between the sim layout (supervisor included,
                         sits between control_node and the bridge) and
                         the real-car layout (supervisor omitted, the
                         on-vehicle uDV handles the same role).
"""
from __future__ import annotations

from typing import Iterable

from launch.actions import EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState

from lifecycle_msgs.msg import Transition


# ---------------------------------------------------------------------
# Topic remappings (IFSSIM /fsds/* surface → pipeline-side names).
# Single source of truth — every node that consumes one of these pulls
# the tuple from here so a remap change can't drift between nodes.
# Mirrors the pre-refactor table that lived in
# docker/dv_pipeline_stack/pipeline.launch.py.
# ---------------------------------------------------------------------
REMAP_LIDAR    = ("/fsds/lidar/Lidar1", "/lidar/Lidar1")
REMAP_GSS      = ("/fsds/gss",          "/gss")
REMAP_IMU      = ("/fsds/imu",          "/imu")
REMAP_GT       = ("/fsds/testing_only/odom", "/testing_only/odom")
REMAP_RPM      = ("/fsds/motor_rpm",    "/motor_rpm")
REMAP_CMD      = ("/fsds/control_command", "/control_command")
REMAP_STEERING = ("/fsds/steering_angle", "/steering_angle")
REMAP_BRAKE    = ("/fsds/brake_pressure", "/brake_pressure")


def auto_active(
    package: str,
    executable: str,
    name: str,
    remappings: Iterable[tuple[str, str]] | None = None,
) -> list:
    """Return [LifecycleNode, configure_event, activate_handler].

    The named node is auto-driven from `unconfigured` to `active` at
    launch start. Used for the management trio whose action/service
    endpoints must be live before mode_manager fans out change_state
    to the autonomy nodes.
    """
    node = LifecycleNode(
        package=package,
        executable=executable,
        name=name,
        namespace="",
        output="screen",
        remappings=list(remappings or []),
    )
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(node),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    # When configure completes (state → 'inactive'), emit activate.
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=node,
        goal_state="inactive",
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ))],
    ))
    return [node, activate, configure]


def autonomy_lifecycle(
    package: str,
    executable: str,
    name: str,
    remappings: Iterable[tuple[str, str]] | None = None,
) -> LifecycleNode:
    """Return a LifecycleNode parked in 'unconfigured'.

    mode_manager drives the configure/activate transitions via
    change_state fan-out when StartMission arrives; until then the
    node holds no subscriptions and emits no traffic.
    """
    return LifecycleNode(
        package=package,
        executable=executable,
        name=name,
        namespace="",
        output="screen",
        remappings=list(remappings or []),
    )


def autonomy_actions() -> list:
    """Pre-baked autonomy lifecycle-node list with their remaps.

    Order in the LaunchDescription doesn't constrain bring-up order;
    mode_manager.AUTONOMY_NODE_ORDER owns that. Listed here in the
    same order for readable `ros2 node list` output.
    """
    return [
        autonomy_lifecycle(
            "odometry_filter_node", "odometry_filter_node",
            "odometry_filter_node",
            remappings=[REMAP_IMU, REMAP_RPM, REMAP_STEERING, REMAP_BRAKE],
        ),
        autonomy_lifecycle(
            "cone_detection", "cone_detection_node", "cone_detection_node",
            remappings=[REMAP_LIDAR],
        ),
        autonomy_lifecycle(
            "cone_slam", "slam_node", "slam_node",
            remappings=[REMAP_IMU, REMAP_RPM, REMAP_GT],
        ),
        autonomy_lifecycle(
            "path_planning", "path_planning_node", "path_planning_node",
        ),
        autonomy_lifecycle(
            "control", "control_node", "control_node",
            # Post-#384 control_node no longer publishes
            # /fsds/control_command directly — its output flows on
            # /ctrl/cmd_internal to mission_control_node, which
            # surfaces it via the RuntimeControl action's Feedback
            # frames for the supervisor (or the uDV on the real car)
            # to relay onto the bridge. No bridge-facing remap needed.
            remappings=[],
        ),
    ]


def management_actions(include_sim_supervisor: bool = True) -> list:
    """Pre-baked management trio, all auto-active.

    Bring-up order: mode_manager first (its activate_mode service must
    be registered before mission_control's StartMission handler tries
    to call into it), mission_control second, sim_supervisor third
    (its StartMission server only opens after mission_control is ready
    to receive its downstream call).

    Args:
        include_sim_supervisor: True for sim/full builds (the
            supervisor sits between control_node and the bridge);
            False for car builds (the on-vehicle uDV replaces it).
    """
    actions: list = []
    actions += auto_active("mode_manager", "mode_manager_node", "mode_manager_node")
    actions += auto_active(
        "mission_control", "mission_control_node", "mission_control_node",
    )
    if include_sim_supervisor:
        # sim_supervisor needs /imu + /motor_rpm + /steering + /brake +
        # /control_command remapped onto /fsds/* so its OdometryFilter
        # (when enabled) and command relay see the bridge's sensor and
        # actuator topics. The IMU/RPM/steering/brake remaps stay even
        # though the C++ odometry_filter_node owns those subscriptions
        # by default (use_external_odometry_filter=True) — the remap
        # is harmless when no subscriber asks for the topic, and lets
        # `--ros-args -p use_external_odometry_filter:=false` fall back
        # to the Python filter without re-adding the remaps.
        actions += auto_active(
            "sim_supervisor", "sim_supervisor_node", "sim_supervisor_node",
            remappings=[REMAP_IMU, REMAP_RPM, REMAP_STEERING, REMAP_BRAKE,
                        REMAP_CMD],
        )
    return actions
