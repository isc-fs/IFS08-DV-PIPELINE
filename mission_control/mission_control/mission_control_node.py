"""
mission_control_node — DV pipeline lifecycle orchestrator (reconciler).

Sits between the uDV (real car) / sim_supervisor (sim uDV emulator) and
the autonomy lifecycle nodes. Post-action-decomposition it speaks ONLY
the stock-typed interface in `interface_contract` — no custom actions —
so the real micro-ROS uDV can be its direct peer with no library
recompile, and the sim runs the *identical* code against the emulator.

Two responsibilities:

  1. **Lifecycle reconciliation.** Reads the uDV AS state machine
     level-triggered off /assi/state (+ the selected mission off
     /ami/mission) and drives mode_manager `activate_mode` so the
     autonomy lifecycle converges to what the AS state demands
     (Ready→configured, Driving→activated, else→torn down). Reports its
     own progress back on /dv/status (the stock stand-in for the old
     SetMission / RuntimeControl action Results — the handshake the uDV
     gates "go" on). Decisions live in the pure, unit-tested
     `reconcile` module; this node owns the ROS plumbing.

  2. **Control-command aggregation + relay.** Aggregates control_node's
     `/ctrl/cmd_internal` (40 Hz fs_msgs/ControlCommand) plus
     `/ctrl/emergency` + `/slam/finished`, and — only while the mission
     is RUNNING — republishes throttle/steering as a normalised
     `geometry_msgs/Twist` on /ctrl/cmd for the uDV/emulator to scale and
     actuate. Emergencies are requested via the uDV's /force_ebs
     (std_srvs/SetBool) service. The autonomy never publishes
     /fsds/control_command directly; this relay is what makes the sim
     path mirror the real-car DVPC→uDV chain.

Liveness: /assi/state is treated as the uDV heartbeat. If it goes stale
(see _ASSI_STALE_S) the pipeline reconciles to torn-down — the watchdog
that replaces the old action goal's implicit connection.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from fs_msgs.msg import ControlCommand
from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import Transition
from std_msgs.msg import Bool, Int32, String, UInt8
from std_srvs.srv import SetBool

from dv_msgs.msg import LifecycleProgress
from dv_msgs.srv import ActivateMode

from mission_control.interface_contract import (
    AS_OFF,
    DV_EMERGENCY,
    DV_FAILED,
    DV_FINISHED,
    DV_PREPARING,
    SERVICE_FORCE_EBS,
    TOPIC_AMI_MISSION,
    TOPIC_ASSI_STATE,
    TOPIC_CTRL_CMD,
    TOPIC_DV_STATUS,
    ami_index_to_mission_id,
)
from mission_control.reconcile import (
    ReconcileAction,
    is_runnable_mission,
    next_action,
    should_request_ebs,
    steady_dv_status,
    target_for,
)

from mode_manager.mode_registry import MISSION_ID_TO_NAME
from mode_manager.mode_manager_node import TRANSITION_SETUP


# #387 — verbs for the /dv/status_msg `stage` string (web spinner).
# Indexed by lifecycle_msgs/Transition ID. Anything outside this map
# falls back to a generic `transitioning(<id>)`.
_TRANSITION_VERB: dict[int, str] = {
    Transition.TRANSITION_CONFIGURE:  "configuring",
    Transition.TRANSITION_ACTIVATE:   "activating",
    Transition.TRANSITION_DEACTIVATE: "deactivating",
    Transition.TRANSITION_CLEANUP:    "cleaning_up",
    TRANSITION_SETUP:                 "setting up",
}

_TRANSITION_PAST: dict[str, str] = {
    "configuring":  "configured",
    "activating":   "activated",
    "deactivating": "deactivated",
    "cleaning_up":  "cleaned_up",
    "setting up":   "set up",
}
_TRANSITION_BARE: dict[str, str] = {
    "configuring":  "configure",
    "activating":   "activate",
    "deactivating": "deactivate",
    "cleaning_up":  "cleanup",
    "setting up":   "setup",
}


def _stage_from_progress(progress) -> str:
    """Render a LifecycleProgress event as a /dv/status_msg stage string.

    Examples:
        configuring cone_detection_node
        activating slam_node
        cone_detection_node configured
        cone_detection_node activate failed: change_state returned …
        slam_node configure skipped

    The verb-first form for `starting` matches "user is waiting for THIS
    step to finish"; the noun-first past-tense form for terminal phases
    matches "THIS step is now in the past". Reads naturally in a
    session-spinner subtitle.
    """
    node = progress.node_name or "?"
    verb = _TRANSITION_VERB.get(
        progress.transition_id, f"transitioning({progress.transition_id})"
    )
    phase = progress.phase
    if phase == "starting":
        return f"{verb} {node}"
    if phase == "ok":
        return f"{node} {_TRANSITION_PAST.get(verb, verb)}"
    if phase == "skipped":
        return f"{node} {_TRANSITION_BARE.get(verb, verb)} skipped"
    if phase in ("failed", "timeout"):
        bare = _TRANSITION_BARE.get(verb, verb)
        suffix = f": {progress.error}" if progress.error else ""
        return f"{node} {bare} {phase}{suffix}"
    return f"{node} {verb} {phase}"


# activate_mode caps at 30 s per autonomy-node transition (Numba JIT for
# cone_detection_node is the hot path). Five nodes × 2 transitions ×
# 30 s worst-case → 240 s headroom; in practice 15-25 s.
_ACTIVATE_MODE_TIMEOUT_S = 240.0
# Cold start: mode_manager (+ ~/setup) can take tens of seconds before
# /activate_mode is discoverable.
_ACTIVATE_MODE_SRV_WAIT_S = 60.0

# Reconcile + /dv/status heartbeat cadence. 10 Hz: well above the >=2 Hz
# the interface contract requires for the heartbeat, and fast enough that
# a go (AS Driving) is acted on within ~100 ms — negligible against the
# multi-second activate it triggers.
_RECONCILE_HZ = 10.0

# /assi/state liveness. If no AS-state heartbeat arrives within this
# window the uDV/emulator (or the link) is considered dead and the
# pipeline reconciles to torn-down. Must be comfortably longer than the
# uDV's publish period (>=2 Hz → 0.5 s) to tolerate jitter.
_ASSI_STALE_S = 1.5


_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# /dv/status + /ctrl/cmd: steady streams, latest-wins. /dv/status is
# latched so a late-joining uDV/emulator immediately sees the last byte.
_STATUS_QOS = _LATCHED_QOS
_CMD_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


class MissionControlNode(LifecycleNode):
    """DV pipeline lifecycle orchestrator (reconciler). See module docstring."""

    NODE_NAME = "mission_control_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        self._mission_id_to_name: dict[int, str] = dict(MISSION_ID_TO_NAME)

        # Reentrant group: the reconcile timer issues async activate_mode
        # calls whose result callbacks must dispatch on the same executor.
        self._cb_group = ReentrantCallbackGroup()

        # --- ROS handles (created in on_configure) ---
        self._activate_mode_client = None
        self._force_ebs_client = None
        self._dv_status_pub = None
        self._dv_status_msg_pub = None
        self._ctrl_cmd_pub = None
        self._reconcile_timer = None
        self._sub_assi = None
        self._sub_ami = None
        self._sub_ctrl_cmd = None
        self._sub_ctrl_emergency = None
        self._sub_slam_finished = None
        self._sub_mode_manager_progress = None

        # --- uDV uplink state ---
        # latest AS-state byte + when it arrived (monotonic) for the
        # staleness watchdog. None until the first heartbeat.
        self._as_state: int | None = None
        self._as_state_stamp: float = 0.0
        self._desired_mission_id: int = 0   # mapped from /ami/mission

        # --- pipeline lifecycle state ---
        self._prepared_mission_id: int = 0  # 0 = nothing configured
        self._activated: bool = False
        self._busy: bool = False            # an activate_mode call in flight
        self._pending_action: ReconcileAction | None = None

        # Sticky terminal/override flags (cleared when torn down / idle).
        self._failed: bool = False
        self._finished: bool = False
        self._emergency: bool = False
        self._ebs_requested: bool = False

        # --- control relay cache ---
        self._latest_ctrl_cmd: ControlCommand = ControlCommand()

        # --- web-spinner progress ---
        self._latest_progress: LifecycleProgress | None = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            "on_configure: creating reconciler I/O + activate_mode client")

        self._activate_mode_client = self.create_client(
            ActivateMode, "/activate_mode", callback_group=self._cb_group)
        # mission_control CALLS the uDV's /force_ebs on emergency.
        self._force_ebs_client = self.create_client(
            SetBool, SERVICE_FORCE_EBS, callback_group=self._cb_group)

        # Downlink to the uDV/emulator.
        self._dv_status_pub = self.create_lifecycle_publisher(
            UInt8, TOPIC_DV_STATUS, _STATUS_QOS)
        self._ctrl_cmd_pub = self.create_lifecycle_publisher(
            Twist, TOPIC_CTRL_CMD, _CMD_QOS)
        # Linux-side diagnostic for the web spinner (the old SetMission
        # feedback `stage`). The uDV ignores this.
        self._dv_status_msg_pub = self.create_lifecycle_publisher(
            String, TOPIC_DV_STATUS + "_msg", 10)

        # Uplink from the uDV/emulator.
        self._sub_assi = self.create_subscription(
            UInt8, TOPIC_ASSI_STATE, self._on_as_state, _LATCHED_QOS,
            callback_group=self._cb_group)
        self._sub_ami = self.create_subscription(
            Int32, TOPIC_AMI_MISSION, self._on_ami_mission, _LATCHED_QOS,
            callback_group=self._cb_group)

        # Control aggregation inputs (latched flags so a late join sees
        # the last-known emergency / finished state immediately).
        self._sub_ctrl_cmd = self.create_subscription(
            ControlCommand, "/ctrl/cmd_internal", self._on_ctrl_cmd, 10,
            callback_group=self._cb_group)
        self._sub_ctrl_emergency = self.create_subscription(
            Bool, "/ctrl/emergency", self._on_ctrl_emergency, _LATCHED_QOS,
            callback_group=self._cb_group)
        self._sub_slam_finished = self.create_subscription(
            Bool, "/slam/finished", self._on_slam_finished, _LATCHED_QOS,
            callback_group=self._cb_group)
        self._sub_mode_manager_progress = self.create_subscription(
            LifecycleProgress, "/mode_manager/progress",
            self._on_mode_manager_progress, 20, callback_group=self._cb_group)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"on_activate: starting reconcile loop ({_RECONCILE_HZ:.0f} Hz)")
        self._reconcile_timer = self.create_timer(
            1.0 / _RECONCILE_HZ, self._tick, callback_group=self._cb_group)
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate: stopping reconcile loop")
        if self._reconcile_timer is not None:
            self.destroy_timer(self._reconcile_timer)
            self._reconcile_timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup: tearing down I/O")
        if self._reconcile_timer is not None:
            self.destroy_timer(self._reconcile_timer)
            self._reconcile_timer = None
        for sub in (self._sub_assi, self._sub_ami, self._sub_ctrl_cmd,
                    self._sub_ctrl_emergency, self._sub_slam_finished,
                    self._sub_mode_manager_progress):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_assi = self._sub_ami = self._sub_ctrl_cmd = None
        self._sub_ctrl_emergency = self._sub_slam_finished = None
        self._sub_mode_manager_progress = None
        self._dv_status_pub = None
        self._dv_status_msg_pub = None
        self._ctrl_cmd_pub = None
        self._activate_mode_client = None
        self._force_ebs_client = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # uplink callbacks
    # ==================================================================
    def _on_as_state(self, msg: UInt8) -> None:
        new = int(msg.data)
        if new != self._as_state:
            self.get_logger().info(f"/assi/state → {new}")
        self._as_state = new
        self._as_state_stamp = time.monotonic()

    def _on_ami_mission(self, msg: Int32) -> None:
        mission_id = ami_index_to_mission_id(int(msg.data))
        if mission_id != self._desired_mission_id:
            self.get_logger().info(
                f"/ami/mission index {msg.data} → registry mission_id "
                f"{mission_id}")
            self._desired_mission_id = mission_id

    def _on_mode_manager_progress(self, msg: LifecycleProgress) -> None:
        self._latest_progress = msg
        if self._dv_status_msg_pub is not None:
            self._dv_status_msg_pub.publish(String(data=_stage_from_progress(msg)))

    # ==================================================================
    # control aggregation callbacks
    # ==================================================================
    def _on_ctrl_cmd(self, msg: ControlCommand) -> None:
        """Cache + (while RUNNING) relay control output as /ctrl/cmd Twist.

        Event-driven: emit on every /ctrl/cmd_internal arrival so the
        relay tracks control_node's tick rate with no added phase — the
        hot path in tight corners. The actuate gate is `_activated`; the
        uDV applies the command only while it is itself in AS Driving, so
        actuation is gated on both ends.
        """
        self._latest_ctrl_cmd = msg
        if self._activated and not (self._emergency or self._finished):
            self._publish_ctrl_cmd()

    def _on_ctrl_emergency(self, msg: Bool) -> None:
        if msg.data and not self._emergency:
            self.get_logger().warn(
                "/ctrl/emergency rising — requesting EBS, stopping command relay")
            self._emergency = True
            self._request_ebs()
            self._publish_dv_status(DV_EMERGENCY)

    def _on_slam_finished(self, msg: Bool) -> None:
        if msg.data and not self._finished:
            self.get_logger().info(
                "/slam/finished rising — mission complete, stopping command relay")
            self._finished = True
            self._publish_dv_status(DV_FINISHED)

    # ==================================================================
    # reconcile loop
    # ==================================================================
    def _tick(self) -> None:
        """Periodic: watchdog, reconcile one step, publish /dv/status."""
        as_state = self._effective_as_state()

        # AS-state-driven EBS (Emergency). The /ctrl/emergency path is
        # handled in its callback; this covers the uDV asserting Emergency.
        if should_request_ebs(as_state):
            self._emergency = True
            self._request_ebs()

        if not self._busy:
            self._reconcile(as_state)

        self._publish_dv_status(self._current_dv_status())

    def _effective_as_state(self) -> int:
        """Latest AS state, or AS_OFF if the heartbeat is stale/absent."""
        if self._as_state is None:
            return AS_OFF
        if (time.monotonic() - self._as_state_stamp) > _ASSI_STALE_S:
            return AS_OFF  # uDV / link dead → tear down (liveness watchdog)
        return self._as_state

    def _reconcile(self, as_state: int) -> None:
        desired = self._desired_mission_id
        target = target_for(as_state, desired)
        action = next_action(
            target, desired, self._prepared_mission_id, self._activated)

        # Clear sticky terminal flags once we are genuinely torn down so a
        # fresh run can report clean status again.
        if (action is ReconcileAction.NONE
                and not self._activated and self._prepared_mission_id == 0):
            self._failed = self._finished = False
            self._emergency = False
            self._ebs_requested = False

        if action is ReconcileAction.NONE:
            return
        if action is ReconcileAction.PREPARE:
            self._call_activate_mode(
                self._mission_id_to_name.get(desired, ""), activate=False,
                action=action, target_mission_id=desired)
        elif action is ReconcileAction.ACTIVATE:
            self._call_activate_mode(
                self._mission_id_to_name.get(self._prepared_mission_id, ""),
                activate=True, action=action,
                target_mission_id=self._prepared_mission_id)
        elif action is ReconcileAction.TEARDOWN:
            self._call_activate_mode(
                "", activate=False, action=action, target_mission_id=0)

    def _call_activate_mode(
        self, mission: str, *, activate: bool,
        action: ReconcileAction, target_mission_id: int,
    ) -> None:
        if self._activate_mode_client is None:
            return
        if not self._activate_mode_client.service_is_ready():
            # mode_manager not up yet; retry on a later tick (don't block).
            return
        self._busy = True
        self._pending_action = action
        if action is ReconcileAction.PREPARE:
            self._failed = False
            self._publish_dv_status(DV_PREPARING)
        self.get_logger().info(
            f"activate_mode(mission={mission!r}, activate={activate}) "
            f"[{action.value}]")
        req = ActivateMode.Request()
        req.mission = mission
        req.activate = activate
        future = self._activate_mode_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_activate_mode_done(f, action, target_mission_id))

    def _on_activate_mode_done(
        self, future, action: ReconcileAction, target_mission_id: int,
    ) -> None:
        self._busy = False
        self._pending_action = None
        resp = None
        try:
            resp = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(f"activate_mode raised: {ex!r}")

        ok = bool(resp is not None and resp.ok)
        if not ok:
            msg = resp.message if resp is not None else "no response"
            self.get_logger().error(
                f"activate_mode [{action.value}] failed: {msg}")
            self._failed = True
            # Leave lifecycle bookkeeping as-is; the next tick re-evaluates
            # (it will retry the same step until AS state changes).
            return

        if action is ReconcileAction.PREPARE:
            self._prepared_mission_id = target_mission_id
            self._activated = False
        elif action is ReconcileAction.ACTIVATE:
            self._activated = True
        elif action is ReconcileAction.TEARDOWN:
            self._prepared_mission_id = 0
            self._activated = False
        self._failed = False

    # ==================================================================
    # /dv/status + /ctrl/cmd + EBS
    # ==================================================================
    def _current_dv_status(self) -> int:
        if self._emergency:
            return DV_EMERGENCY
        if self._finished:
            return DV_FINISHED
        if self._failed:
            return DV_FAILED
        if self._busy and self._pending_action is ReconcileAction.PREPARE:
            return DV_PREPARING
        return steady_dv_status(self._prepared_mission_id, self._activated)

    def _publish_dv_status(self, status: int) -> None:
        if self._dv_status_pub is not None:
            self._dv_status_pub.publish(UInt8(data=int(status)))

    def _publish_ctrl_cmd(self) -> None:
        """Emit one normalised Twist from the cached ControlCommand."""
        if self._ctrl_cmd_pub is None:
            return
        cmd = self._latest_ctrl_cmd
        twist = Twist()
        twist.linear.x = float(cmd.throttle)   # [-1, 1], negative = regen
        twist.angular.z = float(cmd.steering)  # [-1, 1], left positive
        self._ctrl_cmd_pub.publish(twist)

    def _request_ebs(self) -> None:
        if self._ebs_requested:
            return
        self._ebs_requested = True
        if self._force_ebs_client is None or \
                not self._force_ebs_client.service_is_ready():
            self.get_logger().error(
                f"{SERVICE_FORCE_EBS} unavailable — cannot request EBS over "
                "ROS (the uDV should trigger EBS autonomously too)")
            return
        req = SetBool.Request()
        req.data = True
        self.get_logger().warn(f"requesting EBS via {SERVICE_FORCE_EBS}")
        self._force_ebs_client.call_async(req)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
