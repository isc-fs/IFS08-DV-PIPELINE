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
from datetime import datetime

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
from dv_msgs.srv import ActivateMode, StartBag, StopBag

from mission_control.interface_contract import (
    AS_DRIVING,
    AS_OFF,
    AS_READY,
    DV_EMERGENCY,
    DV_FAILED,
    DV_FINISHED,
    DV_IDLE,
    DV_PREPARING,
    DV_READY,
    DV_RUNNING,
    FREE_RUN_MISSION_ID,
    HEARTBEAT_STALE_S,
    SERVICE_FORCE_EBS,
    TOPIC_AMI_MISSION,
    TOPIC_ASSI_STATE,
    TOPIC_CTRL_CMD,
    TOPIC_DV_STATUS,
    ami_index_to_mission_id,
)
from mission_control.interface_qos import UPLINK_QOS
from mission_control.reconcile import (
    ActiveLevel,
    EbsAction,
    ReconcileAction,
    effective_mission_id,
    is_runnable_mission,
    next_action,
    next_ebs_action,
    should_request_ebs,
    target_for,
)

from mode_manager.mode_registry import CONTROL_NODE_NAME, MISSION_ID_TO_NAME
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

# Reconcile / watchdog tick. The /assi/state staleness watchdog is only
# evaluated once per tick, so the tick period is part of the T11.9.4
# detection budget: worst case, the last good heartbeat lands just before
# a tick and the loss is noticed _ASSI_STALE_S + one tick later. 20 Hz
# keeps that at 0.4 + 0.05 = 0.45 s — 50 ms of reaction margin under the
# 0.5 s cap (pinned by test_detection_budget_leaves_reaction_margin) —
# and a go (AS Driving) is acted on within ~50 ms. The margin comes from
# the tick rate, NOT from thinning the 0.4 s window: at the uDV's 10 Hz
# publish cadence the window must stay 4 missed cycles so a couple of
# dropped best-effort samples can't cause a false teardown.
_RECONCILE_HZ = 20.0

# /dv/status heartbeat cadence ON THE WIRE. Deliberately slower than the
# tick: the firmware sizes its DV_STATUS_STALE_MS = 400 ms window as
# "4 missed cycles at 10 Hz" (dv_interface.h), so the wire rate stays
# 10 Hz — _tick publishes every _DV_STATUS_EVERY_N-th tick. Must divide
# _RECONCILE_HZ exactly (test-pinned).
_DV_STATUS_PUB_HZ = 10.0  # reverted: 20Hz congestion-collapsed the uDV micro-ROS link (see IFS08-DV-uDV#166)
_DV_STATUS_EVERY_N = int(_RECONCILE_HZ / _DV_STATUS_PUB_HZ)

# /assi/state liveness. If no AS-state heartbeat arrives within this
# window the uDV/emulator (or the link) is considered dead and the
# pipeline reconciles to torn-down. Sourced from the interface contract
# (HEARTBEAT_STALE_S) so this window and the firmware's DV_STATUS_STALE_MS
# stay a single value; see interface_contract for the rationale.
#
# Detection budget (T11.9.4): worst-case detection = _ASSI_STALE_S + one
# reconcile tick = 0.4 + 0.05 = 0.45 s < 0.5 s cap — see _RECONCILE_HZ
# above. The safety-critical direction (firmware watching /dv/status →
# Emergency/EBS) is independent and faster (~1 ms loop), so that path
# never depended on this budget.
_ASSI_STALE_S = HEARTBEAT_STALE_S


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

        # Free-run flag (default off). When true, the reconciler brings the
        # autonomy floor (everything but control_node) up and records a rosbag
        # whenever the uDV heartbeat is alive, regardless of AS state — for
        # data collection during manual/OFF driving. Read in on_configure so a
        # launch `free_run:=true` override is applied. See interface_contract.
        self.declare_parameter("free_run", False)
        self._free_run: bool = False

        self._mission_id_to_name: dict[int, str] = dict(MISSION_ID_TO_NAME)

        # Reentrant group: the reconcile timer issues async activate_mode
        # calls whose result callbacks must dispatch on the same executor.
        self._cb_group = ReentrantCallbackGroup()

        # --- ROS handles (created in on_configure) ---
        self._activate_mode_client = None
        self._force_ebs_client = None
        self._bag_start_client = None
        self._bag_stop_client = None
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
        # Activation level: NONE / FLOOR (free-run data-collection floor —
        # whole stack up, control logging only) / RUNNING (live run — control
        # reset for the run, /ctrl/cmd relayed). Replaces the old activated bool.
        self._active_level: ActiveLevel = ActiveLevel.NONE
        self._busy: bool = False            # an activate_mode call in flight
        self._pending_action: ReconcileAction | None = None
        self._tick_no: int = 0              # throttles /dv/status to 10 Hz

        # Sticky terminal/override flags (cleared when torn down / idle).
        self._failed: bool = False
        self._finished: bool = False
        self._emergency: bool = False
        self._ebs_requested: bool = False   # set only when /force_ebs acks
        self._ebs_future = None             # in-flight /force_ebs call, if any

        # --- control relay cache ---
        self._latest_ctrl_cmd: ControlCommand = ControlCommand()

        # --- rosbag (free-run auto-recording) ---
        self._bag_active: bool = False   # a recording is running / being started
        self._bag_future = None          # in-flight StartBag call, if any

        # --- web-spinner progress ---
        self._latest_progress: LifecycleProgress | None = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self._free_run = bool(self.get_parameter("free_run").value)
        self.get_logger().info(
            "on_configure: creating reconciler I/O + activate_mode client "
            f"(free_run={self._free_run})")

        self._activate_mode_client = self.create_client(
            ActivateMode, "/activate_mode", callback_group=self._cb_group)
        # mission_control CALLS the uDV's /force_ebs on emergency.
        self._force_ebs_client = self.create_client(
            SetBool, SERVICE_FORCE_EBS, callback_group=self._cb_group)
        # Free-run rosbag auto-recording: mission_control drives the always-on
        # bag_recorder_node (start/stop over these clients). Created regardless
        # of the flag (cheap, idle when free_run is off).
        self._bag_start_client = self.create_client(
            StartBag, "/bag_recorder/start", callback_group=self._cb_group)
        self._bag_stop_client = self.create_client(
            StopBag, "/bag_recorder/stop", callback_group=self._cb_group)

        # Downlink to the uDV/emulator.
        self._dv_status_pub = self.create_lifecycle_publisher(
            UInt8, TOPIC_DV_STATUS, _STATUS_QOS)
        self._ctrl_cmd_pub = self.create_lifecycle_publisher(
            Twist, TOPIC_CTRL_CMD, _CMD_QOS)
        # Linux-side diagnostic for the web spinner (the old SetMission
        # feedback `stage`). The uDV ignores this.
        self._dv_status_msg_pub = self.create_lifecycle_publisher(
            String, TOPIC_DV_STATUS + "_msg", 10)

        # Uplink from the uDV/emulator. BEST_EFFORT/VOLATILE (UPLINK_QOS)
        # to match the firmware's micro-ROS heartbeat idiom — a RELIABLE /
        # TRANSIENT_LOCAL ("latched") reader silently fails to match the
        # uDV's BEST_EFFORT / VOLATILE writer, leaving the reconciler stuck
        # at AS_OFF. Late-join is covered by the steady heartbeat, not by
        # durability. See interface_qos.py.
        self._sub_assi = self.create_subscription(
            UInt8, TOPIC_ASSI_STATE, self._on_as_state, UPLINK_QOS,
            callback_group=self._cb_group)
        self._sub_ami = self.create_subscription(
            Int32, TOPIC_AMI_MISSION, self._on_ami_mission, UPLINK_QOS,
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
        # Finalise any free-run recording so the bag isn't left truncated.
        if self._bag_active:
            self._stop_bag()
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
        self._bag_start_client = None
        self._bag_stop_client = None
        self._bag_future = None
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
        hot path in tight corners. The actuate gate is ActiveLevel.RUNNING
        (a live run) — on the free-run FLOOR control runs and publishes
        /ctrl/cmd_internal (recorded for pilot-vs-autonomy comparison) but
        the level is FLOOR, so nothing is relayed; and the uDV applies the
        command only while in AS Driving, so actuation is gated on both ends.
        """
        self._latest_ctrl_cmd = msg
        if (self._active_level is ActiveLevel.RUNNING
                and not (self._emergency or self._finished)):
            self._publish_ctrl_cmd()

    def _on_ctrl_emergency(self, msg: Bool) -> None:
        # Only honour control's emergency channel during a live run. On the
        # free-run FLOOR control is active (logging) while the human drives —
        # a control-raised emergency there must NOT trip EBS mid-manual-lap.
        # (control never asserts this today; the gate future-proofs it.) The
        # AS-Emergency path in _tick is independent and always honoured.
        if (msg.data and not self._emergency
                and self._active_level is ActiveLevel.RUNNING):
            self.get_logger().warn(
                "/ctrl/emergency rising — requesting EBS, stopping command relay")
            self._emergency = True
            self._request_ebs()
            self._publish_dv_status(DV_EMERGENCY)

    def _on_slam_finished(self, msg: Bool) -> None:
        # Only a live run (ActiveLevel.RUNNING) can "finish". While the
        # free-run FLOOR is mapping, a slam lap-complete must NOT latch
        # FINISHED — the floor is data collection, not a mission.
        if (msg.data and not self._finished
                and self._active_level is ActiveLevel.RUNNING):
            self.get_logger().info(
                "/slam/finished rising — mission complete, stopping command relay")
            self._finished = True
            self._publish_dv_status(DV_FINISHED)

    # ==================================================================
    # reconcile loop
    # ==================================================================
    def _tick(self) -> None:
        """Periodic: watchdog, reconcile one step, publish /dv/status.

        Runs at _RECONCILE_HZ (watchdog detection granularity); the
        /dv/status wire heartbeat is throttled to _DV_STATUS_PUB_HZ so the
        firmware-observed cadence is unchanged by the faster tick.
        """
        self._tick_no += 1
        as_state = self._effective_as_state()

        # AS-state-driven EBS (Emergency). The /ctrl/emergency callback also
        # raises _emergency; either way we (re)issue /force_ebs here every
        # tick until the uDV acks it, so a dropped call is retried, not lost.
        if should_request_ebs(as_state):
            self._emergency = True
        if self._emergency:
            self._request_ebs()

        if not self._busy:
            self._reconcile(as_state)

        # Free-run rosbag: record for the whole session the uDV is powered
        # (heartbeat alive), across OFF / manual / armed alike. Gated ONLY on
        # the flag + heartbeat, never on the AS state, so a manual lap is
        # captured. Stops when the flag is off or the uDV goes away.
        self._update_bag_recording(
            free_run_active=self._free_run and self._heartbeat_alive())

        if self._tick_no % _DV_STATUS_EVERY_N == 0:
            self._publish_dv_status(self._current_dv_status())

    def _heartbeat_alive(self) -> bool:
        """True while the uDV's /assi/state heartbeat is present and fresh.

        This is the "the uDV is powered on" signal the free-run floor gates
        on — distinct from `_effective_as_state`, which folds a stale/absent
        heartbeat into AS_OFF. A dead/absent heartbeat is NOT free-run-active:
        we never run autonomy (or record) against a uDV that isn't there.
        """
        if self._as_state is None:
            return False
        return (time.monotonic() - self._as_state_stamp) <= _ASSI_STALE_S

    def _effective_as_state(self) -> int:
        """Latest AS state, or AS_OFF if the heartbeat is stale/absent."""
        if self._as_state is None:
            return AS_OFF
        if (time.monotonic() - self._as_state_stamp) > _ASSI_STALE_S:
            return AS_OFF  # uDV / link dead → tear down (liveness watchdog)
        return self._as_state

    def _reconcile(self, as_state: int) -> None:
        # Free-run floor is active only while the flag is set AND the uDV is
        # actually present (heartbeat fresh) — never against a dead uDV.
        free_run = self._free_run and self._heartbeat_alive()
        desired = self._desired_mission_id
        target = target_for(as_state, desired, free_run=free_run)
        # Mission to prepare/run: operator selection when runnable, else the
        # free-run fallback (autocross). This is what makes the hand-off warm —
        # the floor already prepared what the driver will arm with.
        eff = effective_mission_id(
            desired, free_run=free_run, free_run_mission_id=FREE_RUN_MISSION_ID)
        action = next_action(
            target, eff, self._prepared_mission_id, self._active_level)

        # Clear sticky terminal flags once we are genuinely torn down and no
        # emergency is still being asserted, so a fresh cycle reports clean
        # status again. Deliberately NOT gated on `action is NONE`: under
        # free-run the floor makes the torn-down action PREPARE (re-raising the
        # floor), so an action==NONE gate would leave a past emergency/finished
        # latched forever. `not should_request_ebs` keeps EBS asserted while
        # the uDV is still in AS Emergency.
        if (self._active_level is ActiveLevel.NONE
                and self._prepared_mission_id == 0
                and not should_request_ebs(as_state)):
            self._failed = self._finished = False
            self._emergency = False
            self._ebs_requested = False
            self._ebs_future = None

        if action is ReconcileAction.NONE:
            return
        if action is ReconcileAction.PREPARE:
            self._call_activate_mode(
                self._mission_id_to_name.get(eff, ""), activate=False,
                action=action, target_mission_id=eff)
        elif action in (ReconcileAction.ACTIVATE,
                        ReconcileAction.ACTIVATE_FLOOR):
            # Bring the whole stack up. Same call for both — the resulting
            # ActiveLevel (RUNNING vs FLOOR) is what gates the /ctrl/cmd relay.
            self._call_activate_mode(
                self._mission_id_to_name.get(self._prepared_mission_id, ""),
                activate=True, action=action,
                target_mission_id=self._prepared_mission_id)
        elif action is ReconcileAction.RESET_CONTROL:
            # Go hand-off: clean-cycle control_node for the run (fresh state);
            # perception/SLAM stay warm.
            self._call_activate_mode(
                self._mission_id_to_name.get(self._prepared_mission_id, ""),
                activate=True, action=action,
                target_mission_id=self._prepared_mission_id,
                reset_nodes=[CONTROL_NODE_NAME])
        elif action is ReconcileAction.TEARDOWN:
            self._call_activate_mode(
                "", activate=False, action=action, target_mission_id=0)

    def _call_activate_mode(
        self, mission: str, *, activate: bool,
        action: ReconcileAction, target_mission_id: int,
        reset_nodes: list[str] | None = None,
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
        reset = list(reset_nodes or [])
        self.get_logger().info(
            f"activate_mode(mission={mission!r}, activate={activate}"
            + (f", reset={reset}" if reset else "")
            + f") [{action.value}]")
        req = ActivateMode.Request()
        req.mission = mission
        req.activate = activate
        req.reset_nodes = reset
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
            self._active_level = ActiveLevel.NONE
        elif action is ReconcileAction.ACTIVATE_FLOOR:
            self._active_level = ActiveLevel.FLOOR
        elif action in (ReconcileAction.ACTIVATE,
                        ReconcileAction.RESET_CONTROL):
            self._active_level = ActiveLevel.RUNNING
        elif action is ReconcileAction.TEARDOWN:
            self._prepared_mission_id = 0
            self._active_level = ActiveLevel.NONE
        self._failed = False

    # ==================================================================
    # /dv/status + /ctrl/cmd + EBS
    # ==================================================================
    def _current_dv_status(self) -> int:
        """The /dv/status byte the uDV consumes for its go-gate + handshake.

        Deliberately keyed off the REAL AS state, NOT the internal active
        level, so the free-run floor is invisible here: while OFF / manual
        the byte stays DV_IDLE (the floor may be actively mapping + recording,
        but the uDV is not in the go-gate and must never see "ready"). Only
        once the driver actually arms does this advertise readiness for the
        operator-SELECTED mission — identical to the non-free-run handshake,
        so free-run can never perturb arm → Ready → Driving.
        """
        if self._emergency:
            return DV_EMERGENCY
        if self._finished:
            return DV_FINISHED
        if self._failed:
            return DV_FAILED

        real_as = self._effective_as_state()
        # Not armed (OFF / manual / stale) → idle handshake regardless of the
        # free-run floor. Also covers standalone missions (Inspection / EBS
        # map to a non-runnable id): the pipeline stays out of them, exactly
        # as when free_run is off.
        if real_as not in (AS_READY, AS_DRIVING):
            return DV_IDLE
        if not is_runnable_mission(self._desired_mission_id):
            return DV_IDLE
        if self._busy and self._pending_action is ReconcileAction.PREPARE:
            return DV_PREPARING
        # Legacy steady mapping, keyed to the DESIRED (operator-selected)
        # mission so the floor's autocross fallback never advertises readiness
        # for a mission the driver didn't pick. Byte-for-byte identical to the
        # pre-free-run handshake for a normal armed run: READY once prepared
        # (on the floor control is up but not reset-for-the-run, so this stays
        # READY, not RUNNING), RUNNING once control has been clean-reset for
        # the run (the firmware holds the launch brakes until it sees RUNNING).
        if self._prepared_mission_id != self._desired_mission_id:
            return DV_IDLE
        if self._active_level is ActiveLevel.RUNNING:
            return DV_RUNNING
        return DV_READY

    def _publish_dv_status(self, status: int) -> None:
        if self._dv_status_pub is not None:
            self._dv_status_pub.publish(UInt8(data=int(status)))

    def _publish_ctrl_cmd(self) -> None:
        """Emit one normalised Twist from the cached ControlCommand."""
        if self._ctrl_cmd_pub is None:
            return
        cmd = self._latest_ctrl_cmd
        twist = Twist()
        # control_node emits SPLIT unsigned channels — throttle[0,1] (motor) and
        # brake[0,1] (regen) — with a deadband guaranteeing only one is non-zero.
        # Pack them into the single signed /ctrl/cmd convention (linear.x ∈
        # [-1, 1], negative = regen) that the uDV / sim_supervisor decode, so
        # throttle − brake is a lossless encode. Sending cmd.throttle ALONE
        # silently dropped the entire regen channel: the car could only coast,
        # never brake — on the bench (wheels on stands) that let a freewheel
        # over-rev trip the inverter because the loop had no braking authority.
        twist.linear.x = float(cmd.throttle) - float(cmd.brake)  # +motor / −regen
        twist.angular.z = float(cmd.steering)  # [-1, 1], left positive
        self._ctrl_cmd_pub.publish(twist)

    def _request_ebs(self) -> None:
        """(Re)issue /force_ebs, retrying until the uDV acknowledges it.

        Driven every reconcile tick while in emergency (see _tick). The
        request is latched done (`_ebs_requested`) ONLY on a positive ack,
        via `_on_ebs_response` — an unavailable service or a failed/negative
        call leaves the path open so the next tick retries. A single dropped
        call can no longer silently kill the EBS request for the session.
        """
        service_ready = (self._force_ebs_client is not None
                         and self._force_ebs_client.service_is_ready())
        call_in_flight = (self._ebs_future is not None
                          and not self._ebs_future.done())
        action = next_ebs_action(
            emergency=self._emergency,
            acked=self._ebs_requested,
            call_in_flight=call_in_flight,
            service_ready=service_ready,
        )
        if action is not EbsAction.DISPATCH:
            if action is EbsAction.WAIT and not service_ready:
                # Throttled: the tick calls us at _RECONCILE_HZ (20 Hz).
                self.get_logger().error(
                    f"{SERVICE_FORCE_EBS} unavailable — cannot request EBS "
                    "over ROS yet, will retry (the uDV should trigger EBS "
                    "autonomously too)", throttle_duration_sec=1.0)
            return

        req = SetBool.Request()
        req.data = True
        self.get_logger().warn(f"requesting EBS via {SERVICE_FORCE_EBS}")
        self._ebs_future = self._force_ebs_client.call_async(req)
        self._ebs_future.add_done_callback(self._on_ebs_response)

    def _on_ebs_response(self, future) -> None:
        """Latch the EBS request done only on a positive ack; else retry."""
        try:
            resp = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(
                f"{SERVICE_FORCE_EBS} call failed: {ex!r} — will retry")
            resp = None
        if resp is not None and resp.success:
            self._ebs_requested = True
            self.get_logger().warn(f"{SERVICE_FORCE_EBS} acknowledged EBS")
        elif resp is not None:
            self.get_logger().error(
                f"{SERVICE_FORCE_EBS} returned not-ok ({resp.message}) — "
                "will retry")
        # Clear the in-flight handle so the next tick can retry if unacked.
        self._ebs_future = None

    # ==================================================================
    # free-run rosbag auto-recording
    # ==================================================================
    def _update_bag_recording(self, *, free_run_active: bool) -> None:
        """Start / stop the free-run rosbag to match `free_run_active`.

        Called every tick. Start is retried idempotently until the recorder
        acks, so a bag_recorder_node that isn't up yet at container start
        doesn't cost the session its recording.
        """
        if free_run_active and not self._bag_active:
            self._start_bag()
        elif not free_run_active and self._bag_active:
            self._stop_bag()

    def _compose_bag_name(self) -> str:
        return f"freerun_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _start_bag(self) -> None:
        if self._bag_start_client is None:
            return
        if self._bag_future is not None and not self._bag_future.done():
            return  # a StartBag is already in flight
        if not self._bag_start_client.service_is_ready():
            self.get_logger().warn(
                "free-run: /bag_recorder/start not up yet — will retry",
                throttle_duration_sec=5.0)
            return
        req = StartBag.Request()
        req.bag_name = self._compose_bag_name()
        # Optimistic latch so we don't fire a duplicate start next tick; a
        # failure ack clears it (see _on_bag_start_done) and the tick retries.
        self._bag_active = True
        self.get_logger().info(f"free-run: starting rosbag {req.bag_name!r}")
        self._bag_future = self._bag_start_client.call_async(req)
        self._bag_future.add_done_callback(self._on_bag_start_done)

    def _on_bag_start_done(self, future) -> None:
        try:
            resp = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(
                f"free-run: StartBag raised: {ex!r} — will retry")
            resp = None
        if resp is not None and resp.ok:
            self.get_logger().info(
                f"free-run: rosbag recording → {resp.bag_path}")
        else:
            err = resp.error if resp is not None else "no response"
            self.get_logger().error(
                f"free-run: rosbag start failed ({err}) — will retry")
            self._bag_active = False   # let the next tick retry
        self._bag_future = None

    def _stop_bag(self) -> None:
        """Finalise the recording. Fire-and-forget — the recorder's StopBag
        is idempotent, so a dropped call just leaves the bag to be closed on
        recorder shutdown."""
        self._bag_active = False
        if self._bag_stop_client is None or not self._bag_stop_client.service_is_ready():
            return
        self.get_logger().info("free-run: stopping rosbag")
        self._bag_stop_client.call_async(StopBag.Request())


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
