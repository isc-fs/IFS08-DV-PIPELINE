"""
mission_control_node — DV pipeline lifecycle orchestrator.

Sits between the supervisor (sim) / uDV (real car) and the autonomy
lifecycle nodes. Two responsibilities:

  1. **Lifecycle orchestration.** Receives SetMission from the
     supervisor and calls mode_manager `activate_mode` with
     activate=false (prepare: ~/setup + configure only, including
     Numba JIT while nodes are inactive). RuntimeControl open calls
     activate_mode with activate=true. Heartbeats relay per-node
     progress from /mode_manager/progress.

  2. **Control-command aggregation + RuntimeControl server.** Post-#384
     (this PR), aggregates control_node's `/ctrl/cmd_internal` topic
     (40 Hz fs_msgs/ControlCommand) plus slam_node's `/slam/finished`
     latched Bool plus control_node's `/ctrl/emergency` latched Bool,
     and surfaces them as RuntimeControl feedback frames to the
     supervisor (which then publishes /fsds/control_command + /signal/ebs
     on the bridge side). The autonomy *never* publishes
     /fsds/control_command directly anymore — this aggregation +
     relay is what makes the sim path mirror the real-car
     DVPC→uDV chain, and the diagram contract explicitly forbids the
     pre-#384 direct path.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from builtin_interfaces.msg import Time as TimeMsg
from fs_msgs.msg import ControlCommand
from lifecycle_msgs.msg import Transition
from std_msgs.msg import Bool

from dv_msgs.action import SetMission, RuntimeControl
from dv_msgs.msg import LifecycleProgress
from dv_msgs.srv import ActivateMode

from mode_manager.mode_registry import (
    MISSION_ID_TO_NAME,
    MISSION_NAME_TO_ID,
    MODE_REGISTRY,
)
from mode_manager.mode_manager_node import TRANSITION_SETUP


# #387 — verbs for the SetMission feedback `stage` string. Indexed
# by lifecycle_msgs/Transition ID. Anything outside this map falls
# back to a generic `transitioning(<id>)`.
#
# TRANSITION_SETUP (=255) is mode_manager's sentinel for the pre-configure
# ~/setup call on each BaseLifecycleNode.
_TRANSITION_VERB: dict[int, str] = {
    Transition.TRANSITION_CONFIGURE:  "configuring",
    Transition.TRANSITION_ACTIVATE:   "activating",
    Transition.TRANSITION_DEACTIVATE: "deactivating",
    Transition.TRANSITION_CLEANUP:    "cleaning_up",
    TRANSITION_SETUP:                 "setting up",
}


# Past-tense + bare forms for each transition verb. Tiny morphology
# table — English-only, fine for a diagnostic string. Don't try to
# derive these from `_TRANSITION_VERB` algorithmically; "configuring"
# stripped of -ing is "configur", not "configure".
_TRANSITION_PAST: dict[str, str] = {
    "configuring":         "configured",
    "activating":          "activated",
    "deactivating":        "deactivated",
    "cleaning_up":         "cleaned_up",
    "setting up": "set up",
}
_TRANSITION_BARE: dict[str, str] = {
    "configuring":  "configure",
    "activating":   "activate",
    "deactivating": "deactivate",
    "cleaning_up":  "cleanup",
    "setting up":   "setup",
}


def _stage_from_progress(progress) -> str:
    """Render a LifecycleProgress event as a SetMission feedback stage.

    Examples:
        configuring cone_detection_node
        activating slam_node
        cone_detection_node configured
        cone_detection_node activate failed: change_state returned …
        slam_node configure skipped

    The verb-first form for `starting` matches "user is waiting for
    THIS step to finish"; the noun-first past-tense form for terminal
    phases matches "THIS step is now in the past". Reads naturally in
    a session-spinner subtitle.
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


# Cap how long we'll wait for mode_manager.activate_mode to respond
# before reporting failure. activate_mode itself caps at 30 s per
# autonomy-node transition (Numba JIT for cone_detection_node is the
# hot path). With five pipeline nodes × 2 transitions each at 30 s worst-case,
# 240 s gives generous headroom; in practice it returns in 15-25 s.
_ACTIVATE_MODE_TIMEOUT_S = 240.0

# Cold start: mode_manager (and downstream ~/setup) can take tens of seconds
# before /activate_mode is discoverable; 5 s was too tight.
_ACTIVATE_MODE_SRV_WAIT_S = 60.0

# Heartbeat cadence while we're blocked waiting on activate_mode. The
# supervisor relays each feedback frame to its own caller (the web
# backend), so this cadence directly drives "is the page progress bar
# visibly moving?" — 0.5 Hz is the floor before users start clicking
# refresh.
_HEARTBEAT_PERIOD_S = 0.5

# RuntimeControl Feedback emission rate (post-#384). 40 Hz matches
# control_node's tick rate — feedback is rate-limited by the slowest
# layer in the chain and there's no benefit to over-emitting. The
# cached topic values that drive each feedback frame are the latest
# arrived; if /ctrl/cmd_internal hasn't ticked yet the previous
# value is reused (controller fail-safes to zero if it hasn't ticked
# at all).
_RUNTIME_CONTROL_FEEDBACK_HZ = 40.0


class MissionControlNode(LifecycleNode):
    """DV pipeline lifecycle orchestrator. See module docstring."""

    NODE_NAME = "mission_control_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # Mission-id ↔ mission-name maps built from MODE_REGISTRY.
        # Kept as instance attrs (not module-level constants) so a
        # future test fixture can swap the registry on a per-instance
        # basis without monkeypatching the module. The maps are
        # SetMission carries mission_id on the wire; ActivateMode still
        # uses the string mode name derived from the registry.
        self._mission_id_to_name: dict[int, str] = dict(MISSION_ID_TO_NAME)
        self._mission_name_to_id: dict[str, int] = dict(MISSION_NAME_TO_ID)
        self.get_logger().info(
            f"mission registry: "
            f"{[(m.mission_id, m.mode_name) for m in MODE_REGISTRY.values()]}"
        )

        self._set_mission_server: ActionServer | None = None
        self._runtime_control_server: ActionServer | None = None
        self._activate_mode_client = None  # mode_manager activate_mode srv

        # Reentrant group: the SetMission action handler blocks on the
        # activate_mode service future, which means another callback —
        # the service response itself — needs to dispatch on the same
        # executor. Default mutually-exclusive groups would deadlock.
        self._cb_group = ReentrantCallbackGroup()

        # RuntimeControl input topics (Phase 2 of the runtime action
        # protocol; #384). control_node pushes throttle/steer to
        # /ctrl/cmd_internal at 40 Hz; the EBS-request signal lives
        # on the latched /ctrl/emergency; slam_node's
        # mission-completion signal lives on the latched
        # /slam/finished. Each callback caches the latest value into
        # an instance variable; the RuntimeControl feedback loop reads
        # them at 40 Hz and forwards to the supervisor.
        self._sub_ctrl_cmd: rclpy.subscription.Subscription | None = None
        self._sub_ctrl_emergency: rclpy.subscription.Subscription | None = None
        self._sub_slam_finished: rclpy.subscription.Subscription | None = None
        # Cached values consumed by the RuntimeControl feedback loop.
        # Initialised to a fail-safe zero command + flags-clear so the
        # supervisor never sees garbage in the brief window between
        # opening the goal and ctrl/slam emitting their first messages.
        self._latest_ctrl_cmd: ControlCommand = ControlCommand()
        self._latest_emergency: bool = False
        self._latest_finished: bool = False

        # #387 — last LifecycleProgress event observed on
        # /mode_manager/progress. Updated by _on_mode_manager_progress
        # from whatever executor thread happens to dispatch the
        # callback; read by the _execute_set_mission wait loop on
        # the action-server thread. Single-attribute assignment is
        # atomic under the GIL, so we don't need a lock here — but we
        # do need a sentinel so the wait loop can tell "no new event
        # since last heartbeat" from "no event ever". The
        # `_progress_seq` counter increments on every event; the wait
        # loop snapshots it and skips its own heartbeat publish when
        # it matches the previous snapshot.
        self._latest_progress: LifecycleProgress | None = None
        self._progress_seq: int = 0
        self._sub_mode_manager_progress: rclpy.subscription.Subscription | None = None

        # Mode name prepared by the last successful SetMission (inactive).
        self._prepared_mission: str | None = None

        # Active RuntimeControl goal — set when _execute_runtime_control
        # begins, cleared on terminate. The /ctrl/cmd_internal
        # subscription callback consults this to know whether to
        # forward each command immediately as a Feedback frame. See
        # the comment block above _on_ctrl_cmd for the latency
        # rationale.
        self._active_runtime_goal_handle = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(
            "on_configure: creating action endpoints + activate_mode client")

        # mode_manager hosts /activate_mode (per its on_configure).
        # Created here so the service is bound by the time a
        # SetMission goal arrives — re-binding inside the action
        # handler would race with the goal acceptance.
        self._activate_mode_client = self.create_client(
            ActivateMode,
            "/activate_mode",
            callback_group=self._cb_group,
        )

        # Phase 1 server. Goal name matches what sim_supervisor's client
        # opens.
        self._set_mission_server = ActionServer(
            self,
            SetMission,
            "~/set_mission",
            execute_callback=self._execute_set_mission,
            callback_group=self._cb_group,
        )

        # Phase 2 server. Supervisor opens this once Phase 1 is ready;
        # we feed it the aggregated control_node + slam_node outputs
        # as RuntimeControl Feedback frames. cancel_callback accepts
        # all cancellations so the supervisor's tear-down path (when
        # mission_control receives mission="" or the operator hits
        # Stop Session) can unblock the feedback loop cleanly.
        self._runtime_control_server = ActionServer(
            self,
            RuntimeControl,
            "~/runtime_control",
            execute_callback=self._execute_runtime_control,
            cancel_callback=lambda _gh: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

        # RuntimeControl input topics. Latched/TRANSIENT_LOCAL on the
        # flags so a late-joining mission_control subscriber gets the
        # last-known emergency / finished state without having to wait
        # for the next publish.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._sub_ctrl_cmd = self.create_subscription(
            ControlCommand, "/ctrl/cmd_internal",
            self._on_ctrl_cmd, 10,
            callback_group=self._cb_group,
        )
        self._sub_ctrl_emergency = self.create_subscription(
            Bool, "/ctrl/emergency",
            self._on_ctrl_emergency, latched,
            callback_group=self._cb_group,
        )
        self._sub_slam_finished = self.create_subscription(
            Bool, "/slam/finished",
            self._on_slam_finished, latched,
            callback_group=self._cb_group,
        )

        # #387 — granular lifecycle progress. mode_manager publishes one
        # LifecycleProgress per change_state call ("starting" before,
        # then "ok"/"skipped"/"failed"/"timeout" after).
        # _execute_set_mission reads the latest event each heartbeat
        # tick and emits a per-node feedback stage instead of the
        # generic "warming_up". Depth 20 ≈ two full activate_mode runs;
        # we'd rather drop than back-pressure mode_manager.
        self._sub_mode_manager_progress = self.create_subscription(
            LifecycleProgress, "/mode_manager/progress",
            self._on_mode_manager_progress, 20,
            callback_group=self._cb_group,
        )

        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # #387 — mode_manager progress callback
    # ------------------------------------------------------------------
    def _on_mode_manager_progress(self, msg: LifecycleProgress) -> None:
        """Cache the latest per-node lifecycle progress event.

        Runs on whatever executor thread dispatches the subscription.
        Both writes (the instance assignments) are individually atomic
        in CPython under the GIL, so the only ordering risk is the
        reader seeing the new event with the old seq counter — we
        sidestep that by writing the message FIRST, then bumping seq.
        The reader checks seq, then reads the message; if it observes
        a fresh seq it's guaranteed to see the matching message.
        """
        self._latest_progress = msg
        self._progress_seq += 1

    # ------------------------------------------------------------------
    # RuntimeControl input topic callbacks
    #
    # Initial impl polled cached values from a 40 Hz timer loop inside
    # _execute_runtime_control. That added ~25 ms of worst-case latency
    # per relay (command lands 1 ms after the tick → waits a full
    # period for the next emission), measurably destabilising
    # Pure Pursuit in tight corners (end-to-end actuator delay went
    # from ~10 ms direct-publish to ~50-65 ms through the action
    # chain — 2-3 control ticks of phase lag). The fix is to emit
    # feedback synchronously from `_on_ctrl_cmd` so the relay tracks
    # control_node's tick rate exactly, with no extra phase. The
    # 40 Hz timer in _execute_runtime_control now only polls the
    # termination flags.
    # ------------------------------------------------------------------
    def _on_ctrl_cmd(self, msg: ControlCommand) -> None:
        self._latest_ctrl_cmd = msg
        # Immediate forward: if a RuntimeControl goal is active,
        # emit the feedback frame now. This is the hot path in tight
        # corners — every microsecond between control_node's publish
        # and the bridge's setCarControls matters at v=3 m/s with
        # κ ≈ 0.5 m⁻¹.
        gh = self._active_runtime_goal_handle
        if gh is not None and gh.is_active:
            self._publish_runtime_feedback(gh)

    def _on_ctrl_emergency(self, msg: Bool) -> None:
        if msg.data and not self._latest_emergency:
            self.get_logger().warn(
                "/ctrl/emergency rising edge — propagating to supervisor "
                "via RuntimeControl Feedback")
        self._latest_emergency = bool(msg.data)

    def _on_slam_finished(self, msg: Bool) -> None:
        if msg.data and not self._latest_finished:
            self.get_logger().info(
                "/slam/finished rising edge — mission complete, "
                "RuntimeControl will close on next tick")
        self._latest_finished = bool(msg.data)

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup")
        for srv in (self._set_mission_server, self._runtime_control_server):
            if srv is not None:
                srv.destroy()
        self._set_mission_server = None
        self._runtime_control_server = None
        self._activate_mode_client = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _execute_set_mission(self, goal_handle):
        """Phase 1 — prepare autonomy for mission_id (0 = tear down)."""
        mission_id = int(goal_handle.request.mission_id)
        if mission_id == 0:
            mission = ""
        else:
            mission = self._mission_id_to_name.get(mission_id)
            if mission is None:
                result = SetMission.Result()
                result.success = False
                result.message = (
                    f"unknown mission_id {mission_id}; valid ids: "
                    f"{sorted(self._mission_id_to_name.keys())} or 0 to tear down"
                )
                self.get_logger().error(result.message)
                goal_handle.abort()
                return result

        mode_log = repr(mission) if mission else "<tear down>"
        self.get_logger().info(
            f"set_mission received: mission_id={mission_id} mode={mode_log}"
        )

        result = SetMission.Result()

        if not self._activate_mode_client.wait_for_service(timeout_sec=_ACTIVATE_MODE_SRV_WAIT_S):
            result.success = False
            result.message = (
                "/activate_mode unavailable; is mode_manager_node active?"
            )
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result

        self._publish_feedback(goal_handle, "configuring")

        req = ActivateMode.Request()
        req.mission = mission
        req.activate = False
        future = self._activate_mode_client.call_async(req)

        deadline = time.monotonic() + _ACTIVATE_MODE_TIMEOUT_S
        next_heartbeat = time.monotonic() + _HEARTBEAT_PERIOD_S
        last_progress_seq_emitted = self._progress_seq

        while not future.done():
            now = time.monotonic()
            if now >= deadline:
                self._activate_mode_client.remove_pending_request(future)
                result.success = False
                result.message = (
                    f"/activate_mode timed out after "
                    f"{_ACTIVATE_MODE_TIMEOUT_S:.0f} s"
                )
                self.get_logger().error(result.message)
                goal_handle.abort()
                return result

            cur_seq = self._progress_seq
            if cur_seq != last_progress_seq_emitted:
                prog = self._latest_progress
                if prog is not None:
                    self._publish_feedback(
                        goal_handle, _stage_from_progress(prog),
                    )
                last_progress_seq_emitted = cur_seq
                next_heartbeat = now + _HEARTBEAT_PERIOD_S

            elif now >= next_heartbeat:
                self._publish_feedback(goal_handle, "warming_up")
                next_heartbeat = now + _HEARTBEAT_PERIOD_S

            time.sleep(0.05)

        srv_resp: ActivateMode.Response = future.result()
        if srv_resp is None:
            result.success = False
            result.message = "/activate_mode returned None (rclpy error)"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result

        result.success = bool(srv_resp.ok)
        result.message = (
            srv_resp.message if srv_resp.message else
            (f"mission_id {mission_id} prepared" if srv_resp.ok
             else f"activate_mode rejected mission_id {mission_id}")
        )
        self._publish_feedback(goal_handle, "ready" if srv_resp.ok else "failed")

        if srv_resp.ok:
            if mission:
                self._prepared_mission = mission
            else:
                self._prepared_mission = None
            goal_handle.succeed()
            self.get_logger().info(
                f"set_mission: mission_id={mission_id} prepared")
        else:
            goal_handle.abort()
            self.get_logger().error(
                f"set_mission: mission_id={mission_id} failed: "
                f"{result.message}")
        return result

    def _publish_feedback(self, goal_handle, stage: str) -> None:
        """Emit one SetMission feedback frame with the given stage."""
        fb = SetMission.Feedback()
        fb.stage = stage
        fb.stamp = self.get_clock().now().to_msg()
        try:
            goal_handle.publish_feedback(fb)
        except Exception as ex:
            # Goal cancelled / aborted out from under us — feedback
            # publish is best-effort.
            self.get_logger().debug(f"feedback publish skipped: {ex}")

    def _execute_runtime_control(self, goal_handle):
        """
        Phase 2 — forward control_node + slam_node outputs to the
        supervisor as RuntimeControl.Feedback frames until the
        mission terminates.

        Feedback emission is *event-driven*: `_on_ctrl_cmd` publishes
        a Feedback frame on every /ctrl/cmd_internal arrival
        (40 Hz, same as control_node's tick rate, zero added phase).
        This handler exists only to poll the termination flags and
        keep the action alive until one fires. The poll cadence is
        fast enough that an emergency/finished raised between ticks
        is serviced within ~_RUNTIME_CONTROL_FEEDBACK_HZ⁻¹ s.

        Termination order (highest → lowest priority):

          1. `goal_handle.is_cancel_requested` — supervisor cancelled
             (operator hit RES, mission switch, tear-down). Result
             outcome="cancelled".
          2. `self._latest_emergency` — control_node raised the EBS
             request (or another node propagated it through
             /ctrl/emergency). Result outcome="emergency"; the final
             feedback frame still carries emergency=true so the
             supervisor knows to publish /signal/ebs.
          3. `self._latest_finished` — slam_node reported mission
             complete (lap-min distance + big-orange). Result
             outcome="finished".

        Previous version emitted feedback synchronously from this
        loop's tick — that added up to 25 ms of phase lag (worst case)
        between control_node's publish and the supervisor's receive,
        which was destabilising Pure Pursuit in tight corners.
        """
        if self._prepared_mission is None:
            result = RuntimeControl.Result()
            result.outcome = "error"
            result.message = (
                "No prepared mission — call SetMission before RuntimeControl"
            )
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result

        if not self._activate_mode_client.wait_for_service(timeout_sec=_ACTIVATE_MODE_SRV_WAIT_S):
            result = RuntimeControl.Result()
            result.outcome = "error"
            result.message = "/activate_mode unavailable"
            goal_handle.abort()
            return result

        act_req = ActivateMode.Request()
        act_req.mission = self._prepared_mission
        act_req.activate = True
        act_future = self._activate_mode_client.call_async(act_req)
        act_deadline = time.monotonic() + _ACTIVATE_MODE_TIMEOUT_S
        while not act_future.done():
            if time.monotonic() >= act_deadline:
                self._activate_mode_client.remove_pending_request(act_future)
                result = RuntimeControl.Result()
                result.outcome = "error"
                result.message = "activate_mode timed out during RuntimeControl open"
                goal_handle.abort()
                return result
            time.sleep(0.05)

        act_resp = act_future.result()
        if act_resp is None or not act_resp.ok:
            result = RuntimeControl.Result()
            result.outcome = "error"
            result.message = (
                act_resp.message if act_resp and act_resp.message
                else "activate_mode failed"
            )
            goal_handle.abort()
            return result

        self.get_logger().info(
            f"runtime_control opened (activated {self._prepared_mission!r})")
        # Publish the seed frame with whatever's in cache so the
        # supervisor doesn't have to wait for the first /ctrl/cmd_internal
        # tick to receive anything. Same fail-safe ControlCommand the
        # cache was initialised with.
        self._active_runtime_goal_handle = goal_handle
        self._publish_runtime_feedback(goal_handle)

        period = 1.0 / _RUNTIME_CONTROL_FEEDBACK_HZ
        result = RuntimeControl.Result()

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    result.outcome = "cancelled"
                    result.message = "supervisor cancelled the goal"
                    self.get_logger().info(
                        "runtime_control: cancelled by supervisor")
                    self._clear_active_if_owner(goal_handle)
                    goal_handle.canceled()
                    return result

                if self._latest_emergency:
                    # Force one last feedback frame with emergency=true
                    # so the supervisor latches /signal/ebs before we
                    # close (the regular _on_ctrl_cmd-driven path may
                    # not fire again before the action closes).
                    self._publish_runtime_feedback(goal_handle)
                    result.outcome = "emergency"
                    result.message = "/ctrl/emergency true"
                    self.get_logger().warn(
                        "runtime_control: terminating on emergency")
                    self._clear_active_if_owner(goal_handle)
                    goal_handle.succeed()
                    return result

                if self._latest_finished:
                    self._publish_runtime_feedback(goal_handle)
                    result.outcome = "finished"
                    result.message = "/slam/finished true"
                    self.get_logger().info(
                        "runtime_control: terminating on finished")
                    self._clear_active_if_owner(goal_handle)
                    goal_handle.succeed()
                    return result

                time.sleep(period)

            # rclpy.ok() turned false → process shutting down. Best
            # effort: succeed with cancelled outcome so the supervisor
            # client doesn't hang on the future.
            result.outcome = "cancelled"
            result.message = "rclpy shutting down"
            self._clear_active_if_owner(goal_handle)
            goal_handle.canceled()
            return result

        except Exception as ex:  # noqa: BLE001 — terminal safety net
            self.get_logger().error(
                f"runtime_control: unexpected error: {ex!r}")
            result.outcome = "error"
            result.message = repr(ex)
            self._clear_active_if_owner(goal_handle)
            try:
                goal_handle.abort()
            except Exception:  # noqa: BLE001
                pass
            return result

    def _clear_active_if_owner(self, goal_handle) -> None:
        """Clear the active-goal pointer only if it still references
        the goal_handle that's terminating. Prevents a stale cancel of
        an old goal from wiping the pointer set by a newly opened goal
        on the mission-switch path: the supervisor's pattern is
        cancel(gh_old) → send_goal(gh_new) in immediate succession,
        and the two execute() coroutines run concurrently on
        MultiThreadedExecutor. Without this guard, gh_old's terminal
        path would null _active_runtime_goal_handle even after
        gh_new's execute() had already pointed it at gh_new, leaving
        _on_ctrl_cmd silently dropping every frame thereafter (the
        symptom: /control_command has no publisher; the car never
        moves)."""
        if self._active_runtime_goal_handle is goal_handle:
            self._active_runtime_goal_handle = None

    def _publish_runtime_feedback(self, goal_handle) -> None:
        """Emit one RuntimeControl feedback frame from cached state."""
        fb = RuntimeControl.Feedback()
        cmd = self._latest_ctrl_cmd
        # fs_msgs/ControlCommand is float64; RuntimeControl.Feedback is
        # float32. Narrowing cast is fine — the wire spec for the real
        # car uDV is float32 too.
        fb.throttle = float(cmd.throttle)
        fb.steering = float(cmd.steering)
        fb.emergency = self._latest_emergency
        fb.finished = self._latest_finished
        fb.stamp = self.get_clock().now().to_msg()
        try:
            goal_handle.publish_feedback(fb)
        except Exception as ex:  # noqa: BLE001
            # Goal closed under us — best-effort just like StartMission.
            self.get_logger().debug(
                f"runtime feedback publish skipped: {ex}")


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
