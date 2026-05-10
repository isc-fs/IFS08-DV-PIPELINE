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

  2. **Control-command forwarding.** Once the mission is running,
     subscribes to the autonomy stack's local control output (from
     control_node) and re-publishes it as RuntimeControl feedback
     onto the supervisor (sim) or directly to the uDV (real car).
     The autonomy *never* publishes /fsds/control_command itself —
     this redirect is what makes the sim path mirror the real-car
     DVPC→uDV chain.

This file is a **lifecycle skeleton**. Action endpoints are stubbed,
mode_manager service call is stubbed, control forwarding is stubbed.
Real wiring lands in step 5.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

from builtin_interfaces.msg import Time as TimeMsg

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
        # we feed it control_node's output as feedback.
        self._runtime_control_server = ActionServer(
            self,
            RuntimeControl,
            "runtime_control",
            execute_callback=self._execute_runtime_control,
            callback_group=self._cb_group,
        )

        return TransitionCallbackReturn.SUCCESS

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
        Phase 2 — forward control commands until finished/emergency
        terminates the action.

        SKELETON: returns immediately with cancelled outcome. Real
        version wires control_node → feedback frames here.
        """
        self.get_logger().info("runtime_control opened (skeleton)")
        result = RuntimeControl.Result()
        result.outcome = "cancelled"
        result.message = "skeleton: not yet wired"
        goal_handle.succeed()
        return result


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
