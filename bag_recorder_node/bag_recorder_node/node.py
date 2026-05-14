"""bag_recorder_node — ROS service host for the Mission Control bag UX (#465).

Hosts two services on the dv_pipeline_stack DDS graph:

  /bag_recorder/start  (dv_msgs/srv/StartBag)
  /bag_recorder/stop   (dv_msgs/srv/StopBag)

Both are thin wrappers over the `recorder` helper module — this file
is the ROS-shaped façade, the actual subprocess management lives in
`recorder.py` so the test suite can drive it without rclpy.

## Why this node exists

The MC backend's `bag_recorder.py` client used to spawn `ros2 bag
record` locally inside mc_backend. That broke for LiDAR + cameras
because mc_backend is forced to FASTDDS_BUILTIN_TRANSPORTS=UDPv4
(SHM cross-container doesn't work for its StartMission action
client). UDP-only subscriber on multi-fragment messages drops 96 %
of /lidar/Lidar1 scans. Running the recorder INSIDE dv_pipeline_stack
puts it in the same DDS context as the publishers — SHM transport,
preallocated histories, no UDP fragmentation. Full 10 Hz LiDAR is
captured cleanly.

## Lifecycle

Always-on. Started from `pipeline.launch.py` alongside `ifssim_bridge`
and `foxglove_bridge`. Not a LifecycleNode — recording is gated by
service calls, not by the autonomy lifecycle, so the operator can
record idle laps or pre-mission scenes too.

## Concurrency

A single recording at a time. `_state_lock` serialises the start /
stop transitions against concurrent service calls (the MC backend
will never call concurrently in practice, but other clients might —
the lock is cheap insurance).
"""
from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.node import Node

from dv_msgs.srv import StartBag, StopBag

from .recorder import (
    BagRecorderError,
    DiskFullError,
    RecorderSpawnError,
    start_recording,
    stop_recording,
)


class BagRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("bag_recorder_node")
        # Active recording state dict (or None when no recording is
        # running). Same shape as `recorder.start_recording` returns —
        # the `_proc`/`_log_fh` fields hold the in-process handles
        # so stop can SIGINT and wait.
        self._active: Optional[dict] = None
        self._lock = threading.Lock()

        self._start_srv = self.create_service(
            StartBag, "/bag_recorder/start", self._on_start,
        )
        self._stop_srv = self.create_service(
            StopBag, "/bag_recorder/stop", self._on_stop,
        )
        self.get_logger().info(
            "bag_recorder_node ready (/bag_recorder/{start,stop})"
        )

    def _on_start(self, request, response):
        with self._lock:
            if self._active and self._active.get("state") == "recording":
                response.ok = False
                response.state = "failed"
                response.bag_path = self._active.get("bag_path", "")
                response.error = (
                    f"another recording already active: "
                    f"{self._active.get('name')}"
                )
                self.get_logger().warning(
                    f"StartBag rejected: {response.error}"
                )
                return response

            bag_name = (request.bag_name or "").strip()
            if not bag_name:
                response.ok = False
                response.state = "failed"
                response.bag_path = ""
                response.error = "bag_name is required"
                return response

            try:
                state = start_recording(bag_name)
            except DiskFullError as ex:
                self._active = None
                response.ok = False
                response.state = "failed"
                response.bag_path = ""
                response.error = str(ex)
                self.get_logger().warning(
                    f"StartBag refused (disk full): {ex}"
                )
                return response
            except RecorderSpawnError as ex:
                self._active = None
                response.ok = False
                response.state = "failed"
                response.bag_path = ""
                response.error = str(ex)
                self.get_logger().error(
                    f"StartBag failed at spawn: {ex}"
                )
                return response
            except BagRecorderError as ex:
                self._active = None
                response.ok = False
                response.state = "failed"
                response.bag_path = ""
                response.error = str(ex)
                self.get_logger().error(
                    f"StartBag generic failure: {ex}"
                )
                return response

            self._active = state
            response.ok = True
            response.state = state.get("state", "recording")
            response.bag_path = state.get("bag_path", "")
            response.error = ""
            self.get_logger().info(
                f"StartBag: recording {state.get('name')} "
                f"(pid={state.get('pid')}) → {state.get('bag_path')}"
            )
            return response

    def _on_stop(self, request, response):
        with self._lock:
            if not self._active or self._active.get("state") in (
                "stopped", "failed", "none",
            ):
                # Idempotent — no error if nothing to stop.
                response.ok = True
                response.state = "none"
                response.bag_path = ""
                response.error = ""
                return response

            name = self._active.get("name")
            try:
                stop_recording(self._active)
            except Exception as ex:
                self._active["state"] = "failed"
                self._active["error"] = str(ex)
                self.get_logger().error(
                    f"StopBag: stop_recording crashed for {name}: {ex}"
                )

            final_state = self._active.get("state", "failed")
            response.ok = (final_state == "stopped")
            response.state = final_state
            response.bag_path = self._active.get("bag_path", "")
            response.error = self._active.get("error", "")
            self.get_logger().info(
                f"StopBag: {name} → {final_state} ({response.bag_path})"
            )
            # Clear the slot so the next StartBag can proceed. Keep
            # the terminal state in a local var for the response, but
            # don't keep the dict alive — its _proc / _log_fh are
            # already closed.
            self._active = None
            return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BagRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
