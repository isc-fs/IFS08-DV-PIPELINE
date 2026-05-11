"""
mission_control_node — DV pipeline lifecycle orchestrator.

Sits between the supervisor (sim) / uDV (real car) and the autonomy
lifecycle nodes. Two responsibilities:

  1. **Lifecycle orchestration.** Receives StartMission from the
     supervisor, calls mode_manager_node's `activate_mode` service
     with the chosen mission flag, mode_manager fans out
     `change_state` calls to each autonomy LifecycleNode (perception,
     slam, path_planning, control). Heartbeats back to the supervisor
     while transitions are in flight; reports ready/failed at the end.

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
from std_msgs.msg import Bool

from dv_msgs.action import StartMission, RuntimeControl
from dv_msgs.srv import ActivateMode


# Cap how long we'll wait for mode_manager.activate_mode to respond
# before reporting failure. activate_mode itself caps at 30 s per
# autonomy-node transition (Numba JIT for cone_detection_node is the
# hot path). With four nodes × 2 transitions each at 30 s worst-case,
# 240 s gives generous headroom; in practice it returns in 15-25 s.
_ACTIVATE_MODE_TIMEOUT_S = 240.0

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

        self._start_mission_server: ActionServer | None = None
        self._runtime_control_server: ActionServer | None = None
        self._activate_mode_client = None  # mode_manager activate_mode srv

        # Reentrant group: the StartMission action handler blocks on the
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
        # StartMission goal arrives — re-binding inside the action
        # handler would race with the goal acceptance.
        self._activate_mode_client = self.create_client(
            ActivateMode,
            "/activate_mode",
            callback_group=self._cb_group,
        )

        # Phase 1 server. Goal name matches what sim_supervisor's client
        # opens.
        self._start_mission_server = ActionServer(
            self,
            StartMission,
            "start_mission_orchestration",
            execute_callback=self._execute_start_mission,
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
            "runtime_control",
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

        return TransitionCallbackReturn.SUCCESS

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
        for srv in (self._start_mission_server, self._runtime_control_server):
            if srv is not None:
                srv.destroy()
        self._start_mission_server = None
        self._runtime_control_server = None
        self._activate_mode_client = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Action handlers (skeletons)
    # ------------------------------------------------------------------
    def _execute_start_mission(self, goal_handle):
        """
        Phase 1 — drive mode_manager through configure/activate of the
        autonomy lifecycle nodes for the requested mission.

        Calls /activate_mode (hosted by mode_manager_node), emits
        periodic feedback heartbeats while the call is in flight, and
        translates the service response into the StartMission action
        result. The mode_manager service itself drives change_state
        on each autonomy LifecycleNode in the AUTONOMY_LIFECYCLE_NODES
        order — this layer is just the action↔service bridge plus
        heartbeat surfacing.
        """
        mission = goal_handle.request.mission
        self.get_logger().info(
            f"start_mission_orchestration received: mission={mission!r}"
        )

        result = StartMission.Result()

        # Wait for /activate_mode to come up. mode_manager is
        # auto-activated at launch, so this should be ~immediate; cap
        # at 5 s to fail fast if the service genuinely isn't there.
        if not self._activate_mode_client.wait_for_service(timeout_sec=5.0):
            result.ready = False
            result.message = (
                "/activate_mode unavailable; is mode_manager_node active?"
            )
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result

        # Initial heartbeat — gives the supervisor (and the web
        # backend behind it) a "we received the goal" signal before
        # the slow activate_mode call begins.
        self._publish_feedback(goal_handle, "configuring")

        # Fire the service. activate_mode internally drives every
        # autonomy LifecycleNode through configure→activate, which
        # includes the cone_detection_node Numba JIT (~10–20 s on
        # Apple Silicon Docker).
        req = ActivateMode.Request()
        req.mission = mission
        future = self._activate_mode_client.call_async(req)

        deadline = time.monotonic() + _ACTIVATE_MODE_TIMEOUT_S
        next_heartbeat = time.monotonic() + _HEARTBEAT_PERIOD_S

        # Wait loop. We can't block on `rclpy.spin_until_future_complete`
        # because we're already inside a callback dispatched by the
        # executor; the standard spin would deadlock. Instead we yield
        # short slices of wall time while the executor (running on
        # other threads of the MultiThreadedExecutor) drains the
        # service response.
        while not future.done():
            now = time.monotonic()
            if now >= deadline:
                self._activate_mode_client.remove_pending_request(future)
                result.ready = False
                result.message = (
                    f"/activate_mode timed out after "
                    f"{_ACTIVATE_MODE_TIMEOUT_S:.0f} s"
                )
                self.get_logger().error(result.message)
                goal_handle.abort()
                return result

            if now >= next_heartbeat:
                # While in flight we don't have visibility into which
                # transition mode_manager is currently driving — for
                # now report a generic "warming_up" stage. Once
                # mode_manager publishes per-node lifecycle progress
                # (follow-up), we can switch on that here.
                self._publish_feedback(goal_handle, "warming_up")
                next_heartbeat = now + _HEARTBEAT_PERIOD_S

            time.sleep(0.05)

        srv_resp: ActivateMode.Response = future.result()
        if srv_resp is None:
            result.ready = False
            result.message = "/activate_mode returned None (rclpy error)"
            self.get_logger().error(result.message)
            goal_handle.abort()
            return result

        result.ready = bool(srv_resp.ok)
        result.message = (
            srv_resp.message if srv_resp.message else
            (f"mission {mission!r} brought up" if srv_resp.ok
             else f"activate_mode rejected mission {mission!r}")
        )
        self._publish_feedback(goal_handle, "ready" if srv_resp.ok else "failed")

        if srv_resp.ok:
            goal_handle.succeed()
            self.get_logger().info(
                f"start_mission_orchestration: mission {mission!r} ready")
        else:
            goal_handle.abort()
            self.get_logger().error(
                f"start_mission_orchestration: mission {mission!r} failed: "
                f"{result.message}")
        return result

    def _publish_feedback(self, goal_handle, stage: str) -> None:
        """Emit one StartMission feedback frame with the given stage."""
        fb = StartMission.Feedback()
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
        self.get_logger().info("runtime_control opened")
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
