"""
car_supervisor_node — on-vehicle mission/actuation adapter.

Replaces sim_supervisor on the real car. sim_supervisor is the action
client that drives mission_control_node in simulation; on the car the
uDV owns the AS state machine and cannot host ROS2 action clients, so
this node bridges the two:

  uDV  ─/assi/state─┐                        ┌─ /steering/cmd ─→ uDV
       ─/ami/mission┘                        ├─ /ctrl/throttle_cmd (placeholder)
                    │                        └─ /force_ebs (srv) ─→ uDV
                    ▼                        ▲
              car_supervisor  ──SetMission──┤
                    │         ──RuntimeControl (feedback: throttle/steering)
                    └──────────→ mission_control_node

Responsibilities:
  * Translate the uDV AS state (/assi/state) + selected mission
    (/ami/mission) into the SetMission (configure) → RuntimeControl
    (activate + run) action protocol on mission_control_node.
  * Relay RuntimeControl feedback (throttle/steering in [-1,1]) to the
    uDV actuation topics, scaled to physical units — ONLY while AS
    Driving (safety invariant).
  * Request EBS via the uDV /force_ebs service on AS Emergency.

The decision logic (which phase, which mission, how to scale) lives in
the pure, unit-tested policy.py / actuation.py. This node is the ROS2
plumbing around them.

⚠️  GAPS (see docs/CAR_ADAPTATION.md):
  * RuntimeControl feedback carries throttle+steering only (no brake);
    the uDV has no throttle/brake actuation sink yet, so throttle is
    published on a placeholder topic. Steering (/steering/cmd) is real.
  * There is no /isc/mission_finished sink on the uDV; mission
    completion currently just closes RuntimeControl + centres steering.
"""
from __future__ import annotations

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import Float32, Int32, UInt8
from std_srvs.srv import SetBool

from dv_msgs.action import RuntimeControl, SetMission

from car_supervisor.actuation import (
    safe_stop_steering_deg,
    steering_norm_to_deg,
    throttle_norm_clamp,
)
from car_supervisor.policy import (
    SupervisorPhase,
    ami_index_to_mission_id,
    is_runnable_mission,
    phase_for_as_state,
    should_actuate,
    should_trigger_ebs,
)


_SENSOR_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
# /assi/state and /ami/mission are reliable, level-triggered (latest
# value matters), so a small reliable latched-ish queue is appropriate.
_STATE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class CarSupervisor(Node):
    """See module docstring."""

    NODE_NAME = "car_supervisor"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # --- Parameters ---
        self._set_mission_action = self.declare_parameter(
            "set_mission_action",
            "/mission_control_node/set_mission").value
        self._runtime_control_action = self.declare_parameter(
            "runtime_control_action",
            "/mission_control_node/runtime_control").value
        self._force_ebs_service = self.declare_parameter(
            "force_ebs_service", "/force_ebs").value
        self._steering_cmd_topic = self.declare_parameter(
            "steering_cmd_topic", "/steering/cmd").value
        self._throttle_cmd_topic = self.declare_parameter(
            "throttle_cmd_topic", "/ctrl/throttle_cmd").value
        # Steering scaling [-1,1] → degrees. PLACEHOLDER default; CONFIRM
        # against the steering geometry before on-track running.
        self._max_steering_deg = float(self.declare_parameter(
            "max_steering_deg", 20.0).value)
        self._steering_safety_limit_deg = float(self.declare_parameter(
            "steering_safety_limit_deg", 25.0).value)

        self._cb_group = ReentrantCallbackGroup()

        # --- State ---
        self._phase: SupervisorPhase = SupervisorPhase.IDLE
        self._ami_mission_id: int = 0          # registry id from /ami/mission
        self._prepared_mission_id: int | None = None
        self._pending_drive: bool = False       # DRIVING seen before prepared
        self._runtime_goal_handle = None
        self._ebs_requested: bool = False

        # --- Action / service clients ---
        self._set_mission_client = ActionClient(
            self, SetMission, self._set_mission_action,
            callback_group=self._cb_group)
        self._runtime_client = ActionClient(
            self, RuntimeControl, self._runtime_control_action,
            callback_group=self._cb_group)
        self._ebs_client = self.create_client(
            SetBool, self._force_ebs_service, callback_group=self._cb_group)

        # --- Actuation publishers ---
        self._steering_pub = self.create_publisher(
            Float32, self._steering_cmd_topic, _SENSOR_QOS)
        self._throttle_pub = self.create_publisher(
            Float32, self._throttle_cmd_topic, _SENSOR_QOS)

        # --- Subscriptions from the uDV ---
        self.create_subscription(
            UInt8, "/assi/state", self._on_as_state, _STATE_QOS,
            callback_group=self._cb_group)
        self.create_subscription(
            Int32, "/ami/mission", self._on_ami_mission, _STATE_QOS,
            callback_group=self._cb_group)

        self.get_logger().info(
            f"car_supervisor up: SetMission={self._set_mission_action}, "
            f"RuntimeControl={self._runtime_control_action}, "
            f"EBS={self._force_ebs_service}, "
            f"steering→{self._steering_cmd_topic} "
            f"(max={self._max_steering_deg}°, "
            f"safety={self._steering_safety_limit_deg}°), "
            f"throttle→{self._throttle_cmd_topic} (placeholder)")
        self.get_logger().warn(
            "car_supervisor actuation: /steering/cmd is real; throttle is a "
            "PLACEHOLDER topic and max_steering_deg is a PLACEHOLDER scale — "
            "confirm before on-track running (docs/CAR_ADAPTATION.md).")

    # ==================================================================
    # uDV inputs
    # ==================================================================
    def _on_ami_mission(self, msg: Int32) -> None:
        mission_id = ami_index_to_mission_id(int(msg.data))
        if mission_id != self._ami_mission_id:
            self.get_logger().info(
                f"/ami/mission index {msg.data} → registry mission_id "
                f"{mission_id}")
            self._ami_mission_id = mission_id

    def _on_as_state(self, msg: UInt8) -> None:
        new_phase = phase_for_as_state(int(msg.data))
        if new_phase is not self._phase:
            self.get_logger().info(
                f"AS state {msg.data} → phase {self._phase.value} → "
                f"{new_phase.value}")
            self._phase = new_phase
            self._enter_phase(new_phase)

    # ==================================================================
    # Phase transitions (side effects). The *policy* lives in policy.py;
    # this just executes the ROS calls for each entered phase.
    # ==================================================================
    def _enter_phase(self, phase: SupervisorPhase) -> None:
        if should_trigger_ebs(phase):
            self._request_ebs()
            self._cancel_runtime()
            self._publish_safe_stop()
            return

        if phase is SupervisorPhase.IDLE:
            # AS Off — tear down any prepared mission, centre wheel.
            self._cancel_runtime()
            if self._prepared_mission_id is not None:
                self._send_set_mission(0)  # 0 = tear down
            self._pending_drive = False
            self._publish_safe_stop()
            return

        if phase is SupervisorPhase.PREPARED:
            # AS Ready — configure the selected mission if runnable.
            self._ebs_requested = False
            if is_runnable_mission(self._ami_mission_id):
                if self._prepared_mission_id != self._ami_mission_id:
                    self._send_set_mission(self._ami_mission_id)
            else:
                self.get_logger().warn(
                    f"AS Ready but AMI mission {self._ami_mission_id} is not "
                    f"a runnable autonomy mission — not preparing")
            self._publish_safe_stop()
            return

        if phase is SupervisorPhase.DRIVING:
            # AS Driving — open RuntimeControl once the mission is ready.
            if self._prepared_mission_id and \
                    is_runnable_mission(self._prepared_mission_id):
                self._open_runtime_control()
            else:
                self.get_logger().warn(
                    "AS Driving but no mission prepared yet — deferring "
                    "RuntimeControl until SetMission completes")
                self._pending_drive = True
            return

        if phase is SupervisorPhase.FINISHED:
            # AS Finished — mission complete; close runtime, centre wheel.
            self._cancel_runtime()
            self._pending_drive = False
            self._publish_safe_stop()
            return

    # ==================================================================
    # SetMission (Phase 1 — configure)
    # ==================================================================
    def _send_set_mission(self, mission_id: int) -> None:
        if not self._set_mission_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn(
                f"{self._set_mission_action} server not available yet; "
                f"will not block — retry on next AS-state change")
            return
        goal = SetMission.Goal()
        goal.mission_id = int(mission_id)
        self.get_logger().info(f"SetMission → mission_id {mission_id}")
        future = self._set_mission_client.send_goal_async(goal)
        future.add_done_callback(
            lambda f, mid=mission_id: self._on_set_mission_goal(f, mid))

    def _on_set_mission_goal(self, future, mission_id: int) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                f"SetMission goal for mission_id {mission_id} rejected")
            return
        goal_handle.get_result_async().add_done_callback(
            lambda f, mid=mission_id: self._on_set_mission_result(f, mid))

    def _on_set_mission_result(self, future, mission_id: int) -> None:
        result = future.result().result
        if result is None or not result.success:
            msg = result.message if result else "no result"
            self.get_logger().error(
                f"SetMission mission_id {mission_id} failed: {msg}")
            self._prepared_mission_id = None
            self._pending_drive = False
            return

        if mission_id == 0:
            self._prepared_mission_id = None
            self.get_logger().info("SetMission tear-down complete")
            return

        self._prepared_mission_id = mission_id
        self.get_logger().info(
            f"SetMission mission_id {mission_id} prepared: {result.message}")
        # If DRIVING arrived while we were still configuring, open the
        # runtime now (only if we are still in DRIVING).
        if self._pending_drive and self._phase is SupervisorPhase.DRIVING:
            self._pending_drive = False
            self._open_runtime_control()

    # ==================================================================
    # RuntimeControl (Phase 2 — activate + run, relay feedback)
    # ==================================================================
    def _open_runtime_control(self) -> None:
        if self._runtime_goal_handle is not None:
            return  # already running
        if not self._runtime_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn(
                f"{self._runtime_control_action} server not available; "
                f"deferring — retry on next AS-state change")
            self._pending_drive = True
            return
        self.get_logger().info("RuntimeControl → opening (activate + run)")
        future = self._runtime_client.send_goal_async(
            RuntimeControl.Goal(),
            feedback_callback=self._on_runtime_feedback)
        future.add_done_callback(self._on_runtime_goal)

    def _on_runtime_goal(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("RuntimeControl goal rejected")
            return
        self._runtime_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            self._on_runtime_result)

    def _on_runtime_feedback(self, feedback_msg) -> None:
        """Relay control output to the uDV — only while AS Driving."""
        if not should_actuate(self._phase):
            # State changed out from under the action (e.g. EMERGENCY);
            # do not actuate. The phase-entry handler already issued the
            # safe stop.
            return
        fb = feedback_msg.feedback
        try:
            steering_deg = steering_norm_to_deg(
                float(fb.steering),
                max_steering_deg=self._max_steering_deg,
                safety_limit_deg=self._steering_safety_limit_deg)
        except ValueError as ex:
            self.get_logger().warn(
                f"steering scale skipped ({ex}); centring wheel")
            steering_deg = safe_stop_steering_deg()
        self._steering_pub.publish(Float32(data=float(steering_deg)))
        self._throttle_pub.publish(
            Float32(data=throttle_norm_clamp(float(fb.throttle))))

    def _on_runtime_result(self, future) -> None:
        result = future.result().result
        outcome = result.outcome if result else "unknown"
        self.get_logger().info(f"RuntimeControl closed: outcome={outcome}")
        self._runtime_goal_handle = None
        self._publish_safe_stop()
        # An emergency outcome means the pipeline raised EBS — escalate.
        if outcome == "emergency":
            self._request_ebs()

    def _cancel_runtime(self) -> None:
        gh = self._runtime_goal_handle
        if gh is None:
            return
        self.get_logger().info("RuntimeControl → cancelling")
        gh.cancel_goal_async()
        self._runtime_goal_handle = None

    # ==================================================================
    # EBS + safe stop
    # ==================================================================
    def _request_ebs(self) -> None:
        if self._ebs_requested:
            return
        self._ebs_requested = True
        if not self._ebs_client.wait_for_service(timeout_sec=0.0):
            self.get_logger().error(
                f"{self._force_ebs_service} unavailable — cannot request EBS "
                f"over ROS (the uDV should trigger EBS autonomously too)")
            return
        req = SetBool.Request()
        req.data = True
        self.get_logger().warn(f"requesting EBS via {self._force_ebs_service}")
        self._ebs_client.call_async(req)

    def _publish_safe_stop(self) -> None:
        """Centre the wheel and zero throttle (non-driving phases)."""
        self._steering_pub.publish(
            Float32(data=safe_stop_steering_deg()))
        self._throttle_pub.publish(Float32(data=0.0))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarSupervisor()
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
