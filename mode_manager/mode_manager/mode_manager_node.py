"""
mode_manager_node — lifecycle fan-out for the autonomy stack.

Hosts one service: `activate_mode` (dv_msgs/ActivateMode). On request
it drives `change_state` for each autonomy LifecycleNode (perception,
slam, path_planning, control), propagating the mission flag so each
node activates with the correct strategy.

Lifecycle of mode_manager itself is trivial — it stays `active` for
the whole container lifetime; it's the autonomy nodes underneath that
move between unconfigured → inactive → active.

Skeleton: the activate_mode handler returns ok=true without touching
any change_state services. Real wiring lands in step 5.
"""

from __future__ import annotations

import time
from typing import Iterable

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

from lifecycle_msgs.msg import State as LifecycleStateMsg, Transition
from lifecycle_msgs.srv import ChangeState, GetState

from dv_msgs.msg import LifecycleProgress
from dv_msgs.srv import ActivateMode


# Mapping from a lifecycle transition ID → the state ID it lands in.
# Used by _drive_transition's pre-check: if the node is already in
# the target state (or past it on the path we're walking), skip the
# change_state call. Makes activate_mode idempotent — calling
# activate_mode("trackdrive") twice in a row, or activate_mode("")
# on an already-unconfigured stack, succeeds without invalid-
# transition errors.
_TRANSITION_TARGET_STATE: dict[int, int] = {
    Transition.TRANSITION_CONFIGURE:  LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
    Transition.TRANSITION_ACTIVATE:   LifecycleStateMsg.PRIMARY_STATE_ACTIVE,
    Transition.TRANSITION_DEACTIVATE: LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
    Transition.TRANSITION_CLEANUP:    LifecycleStateMsg.PRIMARY_STATE_UNCONFIGURED,
}

# Per bring-up / tear-down direction, the set of states from which a
# transition is a no-op (we're already at or past the goal).
#
# bring_up (CONFIGURE then ACTIVATE):
#   • CONFIGURE: skip if already in inactive or active
#   • ACTIVATE:  skip if already in active
# tear_down (DEACTIVATE then CLEANUP):
#   • DEACTIVATE: skip if already in inactive or unconfigured
#   • CLEANUP:    skip if already in unconfigured
_TRANSITION_SKIP_STATES: dict[int, frozenset[int]] = {
    Transition.TRANSITION_CONFIGURE: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
        LifecycleStateMsg.PRIMARY_STATE_ACTIVE,
    }),
    Transition.TRANSITION_ACTIVATE: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_ACTIVE,
    }),
    Transition.TRANSITION_DEACTIVATE: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
        LifecycleStateMsg.PRIMARY_STATE_UNCONFIGURED,
    }),
    Transition.TRANSITION_CLEANUP: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_UNCONFIGURED,
    }),
}


# Names of the autonomy LifecycleNodes that mode_manager fans out to.
# Order matters when bringing up: perception before slam before
# planning before control. Reverse order on tear-down.
#
# `lifecycle_managed` flags which entries are real LifecycleNodes today
# vs. legacy plain Nodes that ignore change_state calls. Entries with
# `lifecycle_managed=False` get logged but skipped — once a node is
# converted, flip the flag and mode_manager starts driving its
# transitions.
AUTONOMY_LIFECYCLE_NODES: tuple[tuple[str, bool], ...] = (
    ("cone_detection_node", True),
    ("slam_node",            True),
    ("path_planning_node",   True),
    ("control_node",         True),
)

# Per-service-call timeout when waiting for a change_state response.
# The expensive transition is on_configure for cone_detection_node
# (Numba JIT, ~10-20 s); 30 s gives generous headroom on Apple
# Silicon Docker hosts where JIT is slower than bare Linux.
_CHANGE_STATE_TIMEOUT_S: float = 30.0

VALID_MISSIONS: frozenset[str] = frozenset(
    {"trackdrive", "autocross", "accel", "skidpad"}
)


class ModeManagerNode(LifecycleNode):
    """Lifecycle fan-out service host. See module docstring."""

    NODE_NAME = "mode_manager_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._activate_mode_srv = None
        # change_state clients keyed by node name. Lazy-created in
        # on_configure so the autonomy nodes have time to register
        # their lifecycle services before we try to bind to them.
        self._change_state_clients: dict[str, rclpy.client.Client] = {}
        # #387 — per-node-per-transition progress publisher. One event
        # per change_state call ("starting" before, "ok"/"failed"/
        # "skipped"/"timeout" after) so mission_control_node can echo
        # the current step into StartMission feedback. Created in
        # on_configure alongside the lifecycle clients; published
        # to / unconditionally (cheap event-driven topic, no
        # subscribers → no-op).
        self._progress_pub = None
        # Reentrant group so the inner spin_until_future_complete inside
        # _drive_transition can dispatch the change_state response while
        # the activate_mode service callback that triggered it is still
        # on the call stack. Without this the default mutually-exclusive
        # group would deadlock.
        self._cb_group = ReentrantCallbackGroup()
        # get_state clients for the pre-check that makes _fan_out
        # idempotent (skip transitions whose target state is already
        # reached). Lazy-created in on_configure alongside the
        # change_state clients.
        self._get_state_clients: dict[str, rclpy.client.Client] = {}

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_configure: creating activate_mode service")
        self._activate_mode_srv = self.create_service(
            ActivateMode,
            "activate_mode",
            self._handle_activate_mode,
            callback_group=self._cb_group,
        )
        # #387 — progress topic. Depth=10 KEEP_LAST: bursty during the
        # ~4-6 transitions of a single activate_mode call, idle the
        # rest of the time. A late-joining subscriber misses the most
        # recent burst but that's fine — mission_control_node's
        # heartbeat fills the gap. Default RELIABLE is correct (small
        # payload, drops would defeat the point).
        self._progress_pub = self.create_publisher(
            LifecycleProgress, "/mode_manager/progress", 10,
        )
        # One change_state + one get_state client per managed autonomy
        # node. ROS will not error if the server isn't up yet;
        # wait_for_service is used at call-time inside the helpers.
        for name, managed in AUTONOMY_LIFECYCLE_NODES:
            if not managed:
                continue
            self._change_state_clients[name] = self.create_client(
                ChangeState,
                f"/{name}/change_state",
                callback_group=self._cb_group,
            )
            self._get_state_clients[name] = self.create_client(
                GetState,
                f"/{name}/get_state",
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
        if self._activate_mode_srv is not None:
            self.destroy_service(self._activate_mode_srv)
        self._activate_mode_srv = None
        if self._progress_pub is not None:
            self.destroy_publisher(self._progress_pub)
            self._progress_pub = None
        for cli in self._change_state_clients.values():
            self.destroy_client(cli)
        self._change_state_clients.clear()
        for cli in self._get_state_clients.values():
            self.destroy_client(cli)
        self._get_state_clients.clear()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------
    def _handle_activate_mode(
        self,
        request: ActivateMode.Request,
        response: ActivateMode.Response,
    ) -> ActivateMode.Response:
        """
        Drive change_state across every managed autonomy node.

        Bring-up order: forward through AUTONOMY_LIFECYCLE_NODES with
        CONFIGURE then ACTIVATE.
        Tear-down order: reverse, with DEACTIVATE then CLEANUP.

        Nodes flagged `lifecycle_managed=False` are logged-and-skipped
        (the legacy plain-Node ones still under conversion). Once they
        flip to True, mode_manager picks them up automatically.

        Mission flag propagation (parameter pre-set on each node) is
        not yet wired — first iteration only validates the lifecycle
        mechanism. Mission-strategy parameter push lands once the
        autonomy nodes start consuming it.
        """
        mission = request.mission

        if mission == "":
            self.get_logger().info("activate_mode: tearing down autonomy")
            ok, msg = self._fan_out(
                reversed(AUTONOMY_LIFECYCLE_NODES),
                (Transition.TRANSITION_DEACTIVATE, Transition.TRANSITION_CLEANUP),
                "tear_down",
            )
            response.ok = ok
            response.message = msg
            return response

        if mission not in VALID_MISSIONS:
            response.ok = False
            response.message = (
                f"unknown mission {mission!r}; must be one of "
                f"{sorted(VALID_MISSIONS)}"
            )
            self.get_logger().warning(response.message)
            return response

        self.get_logger().info(
            f"activate_mode: bringing up autonomy for mission={mission!r}"
        )
        ok, msg = self._fan_out(
            AUTONOMY_LIFECYCLE_NODES,
            (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE),
            "bring_up",
        )
        response.ok = ok
        response.message = msg if not ok else f"brought up for {mission!r}"
        return response

    # ------------------------------------------------------------------
    # Lifecycle fan-out
    # ------------------------------------------------------------------
    def _fan_out(
        self,
        nodes: Iterable[tuple[str, bool]],
        transitions: tuple[int, ...],
        phase: str,
    ) -> tuple[bool, str]:
        """
        Drive each managed node through `transitions` in order. Stops
        on the first failure and returns (False, diagnostic). Skipped
        unmanaged nodes are noted in the log but don't fail the call.
        """
        for name, managed in nodes:
            if not managed:
                self.get_logger().info(
                    f"  [{phase}] {name}: not lifecycle-managed yet, skipping"
                )
                continue
            for t in transitions:
                ok, err = self._drive_transition(name, t)
                if not ok:
                    return False, f"{name}: {err}"
        return True, ""

    def _drive_transition(self, node_name: str, transition_id: int) -> tuple[bool, str]:
        """Call /<node_name>/change_state with `transition_id`. Blocks.

        Idempotency pre-check: if the node is already in (or past)
        the transition's target state, return success without calling
        change_state. Lets activate_mode("trackdrive") be invoked
        repeatedly without invalid-transition errors, and lets
        activate_mode("") succeed on an already-unconfigured stack
        (the common case at fresh-container startup).

        Publishes /mode_manager/progress events ("starting", then one
        of "ok"/"failed"/"timeout"/"skipped") so mission_control_node
        can echo per-node progress into StartMission feedback (#387).
        """
        cli = self._change_state_clients.get(node_name)
        if cli is None:
            self._publish_progress(node_name, transition_id, "failed",
                                   "no change_state client (not in managed set)")
            return False, "no change_state client (not in managed set)"

        # State-check: skip if already at/past target. The get_state
        # call is cheap (~ms) and removes the need for callers to
        # track autonomy state externally.
        current = self._get_current_state(node_name)
        if current is None:
            # get_state failed — proceed to attempt the transition;
            # the change_state error will be more informative than
            # bailing here.
            self.get_logger().debug(
                f"  [{node_name}] get_state unavailable; "
                f"attempting transition {transition_id} blind"
            )
        else:
            skip_states = _TRANSITION_SKIP_STATES.get(transition_id, frozenset())
            if current in skip_states:
                self.get_logger().info(
                    f"  [{node_name}] already at/past target for "
                    f"transition {transition_id} (state={current}); skipping"
                )
                self._publish_progress(node_name, transition_id, "skipped", "")
                return True, ""

        # Announce we're about to start. Subscribers can latch this and
        # surface "configuring <node>" before the (potentially long)
        # change_state call returns — that's the whole point of #387.
        self._publish_progress(node_name, transition_id, "starting", "")

        if not cli.wait_for_service(timeout_sec=_CHANGE_STATE_TIMEOUT_S):
            err = (
                f"/{node_name}/change_state did not appear within "
                f"{_CHANGE_STATE_TIMEOUT_S}s"
            )
            self._publish_progress(node_name, transition_id, "timeout", err)
            return False, err

        req = ChangeState.Request()
        req.transition.id = transition_id
        future = cli.call_async(req)
        # Sleep-poll until the future resolves. We can't use
        # rclpy.spin_until_future_complete here because we're already
        # inside a callback dispatched by this node's outer
        # MultiThreadedExecutor; a nested spin would create a second
        # executor and try to re-attach the same node, which deadlocks
        # under default settings. The ReentrantCallbackGroup that owns
        # this client lets the outer MTE dispatch the response while
        # we yield wall time below — same pattern as
        # mission_control_node._execute_start_mission.
        deadline = time.monotonic() + _CHANGE_STATE_TIMEOUT_S
        while not future.done():
            if time.monotonic() >= deadline:
                cli.remove_pending_request(future)
                err = f"timeout waiting for transition {transition_id}"
                self._publish_progress(node_name, transition_id, "timeout", err)
                return False, err
            time.sleep(0.05)
        result = future.result()
        if result is None:
            err = "change_state returned None (rclpy error)"
            self._publish_progress(node_name, transition_id, "failed", err)
            return False, err
        if not result.success:
            err = f"transition {transition_id} returned success=False"
            self._publish_progress(node_name, transition_id, "failed", err)
            return False, err
        self.get_logger().info(
            f"  [{node_name}] transition {transition_id} ok"
        )
        self._publish_progress(node_name, transition_id, "ok", "")
        return True, ""

    def _publish_progress(
        self,
        node_name: str,
        transition_id: int,
        phase: str,
        error: str,
    ) -> None:
        """Emit one LifecycleProgress event. Best-effort — publish
        failures are logged at debug and never bubble up; the lifecycle
        fan-out itself is the source of truth, the progress topic is
        diagnostic-grade only.
        """
        if self._progress_pub is None:
            return
        msg = LifecycleProgress()
        msg.node_name = node_name
        msg.transition_id = int(transition_id)
        msg.phase = phase
        msg.error = error
        msg.stamp = self.get_clock().now().to_msg()
        try:
            self._progress_pub.publish(msg)
        except Exception as ex:
            self.get_logger().debug(
                f"_publish_progress: publish failed for "
                f"{node_name}/{transition_id}/{phase}: {ex}"
            )

    def _get_current_state(self, node_name: str) -> int | None:
        """Query /<node_name>/get_state. Returns the state ID (an int
        from lifecycle_msgs.msg.State.PRIMARY_STATE_*), or None if the
        service is unavailable or the call timed out.

        Short timeout: this is a fast intra-container service call,
        and we use it from inside _drive_transition's hot path. If
        it doesn't respond in 2 s the node is probably stuck and
        the change_state call will fail with a more informative
        error anyway.
        """
        cli = self._get_state_clients.get(node_name)
        if cli is None:
            return None
        if not cli.service_is_ready():
            # No wait — if the service isn't already there, we won't
            # wait for it. The downstream change_state will gate on
            # wait_for_service.
            return None

        future = cli.call_async(GetState.Request())
        deadline = time.monotonic() + 2.0
        while not future.done():
            if time.monotonic() >= deadline:
                cli.remove_pending_request(future)
                return None
            time.sleep(0.02)
        result = future.result()
        if result is None:
            return None
        return int(result.current_state.id)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ModeManagerNode()
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
