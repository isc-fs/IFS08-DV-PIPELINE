#!/usr/bin/env python3
"""Offline replay harness for cone_graph_slam (P2 of issue #282).

Reads a rosbag2 capture (`/imu` + `/Conos_raw` + `/motor_rpm` +
`/testing_only/odom`) and drives `ConeGraphSlamNode` through it
deterministically. Compares SLAM pose to GT (re-anchored to SLAM's
calibration-end frame, the same way the live diagnostic does) and
emits per-scan residuals plus a pass/fail summary.

This is the gate every cone_slam-touching PR has to pass before
merging onto dev. Catches IndeterminantLinearSystemException, runaway
drift, DA cascades, all without spinning UE5.

## Capturing a bag

Inside the container, while the live pipeline is running:

    docker compose exec dv_pipeline_stack bash -lc \
      'source /opt/ros/humble/setup.bash && \
       ros2 bag record -o /tmp/slam_bag \
         /imu /Conos_raw /motor_rpm /testing_only/odom'

Then drive a representative session, Ctrl-C the recorder when done.
Copy the bag out if you want it on the host:

    docker compose cp dv_pipeline_stack:/tmp/slam_bag ./slam_bag

## Running the replay

    docker compose exec -T dv_pipeline_stack bash -lc \
      'source /opt/ros/humble/setup.bash && \
       source /dv_pipeline_stack_ws/install/setup.bash && \
       python3 /dv_pipeline_stack_ws/src/cone_slam/scripts/replay_slam.py \
         /tmp/slam_bag --csv /tmp/residuals.csv'

Exit 0  = pose-vs-GT residual stayed within `--max-pose-err-m` and
          SLAM never raised.
Exit 1  = SLAM crashed mid-replay OR exceeded the residual threshold.
Exit 2  = bag missing required topics / file not found.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np


_LEGACY_REQUIRED_TOPICS = {
    "/imu",
    "/Conos_raw",
    "/motor_rpm",
    "/testing_only/odom",
}


def _open_bag(path: str):
    # Imports deferred so `--help` works outside the container.
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    # Auto-detect storage_id from the bag's metadata.yaml so we work
    # against both sqlite3 (older capture path) and mcap (post-#465).
    storage_id = "sqlite3"
    meta = Path(path) / "metadata.yaml"
    if meta.exists():
        for line in meta.read_text().splitlines():
            line = line.strip()
            if line.startswith("storage_identifier:"):
                storage_id = line.split(":", 1)[1].strip()
                break
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id=storage_id),
        ConverterOptions("", ""),
    )
    return reader


def _topic_msg_classes(reader) -> dict:
    from rosidl_runtime_py.utilities import get_message
    out = {}
    for tt in reader.get_all_topics_and_types():
        out[tt.name] = get_message(tt.type)
    return out


def _odom_to_pose3(msg):
    """Match `cone_graph_slam_node._odom_to_pose3`."""
    import gtsam
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z),
        np.array([p.x, p.y, p.z]),
    )


def _yaw_err(slam_yaw: float, gt_yaw: float) -> float:
    """Wrap to [-pi, pi]."""
    d = slam_yaw - gt_yaw
    return float(np.arctan2(np.sin(d), np.cos(d)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline replay of cone_slam against a rosbag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("bag", help="path to rosbag2 directory")
    ap.add_argument(
        "--max-pose-err-m", type=float, default=1.5,
        help="fail if SLAM-vs-GT position residual exceeds this (default 1.5 m)")
    ap.add_argument(
        "--max-yaw-err-deg", type=float, default=15.0,
        help="fail if SLAM-vs-GT yaw residual exceeds this (default 15°)")
    ap.add_argument(
        "--csv", type=str, default=None,
        help="dump per-scan residuals to this CSV path")
    ap.add_argument(
        "--max-scans", type=int, default=0,
        help="stop after N cone scans (default 0 = no limit)")
    ap.add_argument(
        "--quiet", action="store_true",
        help="suppress per-scan progress lines")
    ap.add_argument(
        "--veto-m", type=float, default=None,
        help="override `new_landmark_proximity_veto_m`; pass 0 to "
             "disable the cascade-guard veto for baseline comparison")
    ap.add_argument(
        "--node-class", type=str,
        default="cone_slam.cone_graph_slam_node:ConeGraphSlamNode",
        help="SLAM node class to drive, as `module.path:ClassName`. "
             "Default targets the current cone_graph_slam. Switch to "
             "the new class once the rewrite lands "
             "(e.g. `cone_slam.slam_node:SlamNode`).")
    ap.add_argument(
        "--behavior", type=str, default="autocross",
        help="Behavior name passed to BaseLifecycleNode-style nodes "
             "(mode_manager Setup contract). Ignored by the legacy "
             "cone_graph node. Default: autocross.")
    ap.add_argument(
        "--pose-source", type=str, default="gt",
        help="Pose feed for the new SlamNode (odom|gt). Default 'gt' "
             "uses /testing_only/odom as the pose source — the right "
             "setting for Phase 1 baselines that don't want /odom "
             "drift dominating the residual. Ignored by the legacy "
             "node.")
    ap.add_argument(
        "--param", action="append", default=[],
        metavar="NAME=VALUE",
        help="Set an arbitrary ROS parameter on the SLAM node before "
             "configure. Repeatable. Example: --param "
             "phase2_force_freeze_after_n_scans=200. Value is parsed "
             "as int, then float, then string.")
    args = ap.parse_args()

    if not Path(args.bag).exists():
        print(f"bag not found: {args.bag}", file=sys.stderr)
        return 2

    reader = _open_bag(args.bag)
    topic_classes = _topic_msg_classes(reader)

    # Heavy ROS imports past the arg-parse / file-presence checks.
    import rclpy
    from rclpy.serialization import deserialize_message

    rclpy.init()
    node = None
    try:
        # Dynamic import — class-agnostic so we can swap in the new
        # rewrite class without editing this script. Format is
        # `module.path:ClassName`.
        try:
            mod_name, _, cls_name = args.node_class.partition(":")
            if not mod_name or not cls_name:
                raise ValueError("--node-class must be `module.path:ClassName`")
            import importlib
            slam_mod = importlib.import_module(mod_name)
            SlamNodeCls = getattr(slam_mod, cls_name)
        except (ImportError, AttributeError, ValueError) as ex:
            print(f"--node-class {args.node_class!r} failed to resolve: {ex}",
                  file=sys.stderr)
            return 2

        node = SlamNodeCls()

        # Detect node interface: new SlamNode exposes replay_snapshot /
        # replay_dispatch / REPLAY_TOPICS; legacy ConeGraphSlamNode
        # does not. Switch behavior accordingly.
        is_new_node = hasattr(node, "replay_snapshot")
        required_topics = (
            set(node.REPLAY_TOPICS) if is_new_node
            else _LEGACY_REQUIRED_TOPICS
        )
        missing = required_topics - set(topic_classes.keys())
        if missing:
            print(f"bag missing required topics: {sorted(missing)}",
                  file=sys.stderr)
            print(f"  found topics: {sorted(topic_classes.keys())}",
                  file=sys.stderr)
            return 2

        from rclpy.parameter import Parameter

        def _coerce(v: str):
            try:
                return int(v), Parameter.Type.INTEGER
            except ValueError:
                pass
            try:
                return float(v), Parameter.Type.DOUBLE
            except ValueError:
                pass
            return v, Parameter.Type.STRING

        extra_params: list[Parameter] = []
        for kv in args.param:
            if "=" not in kv:
                print(f"--param expects NAME=VALUE, got {kv!r}",
                      file=sys.stderr)
                return 2
            name, _, val = kv.partition("=")
            cval, ctype = _coerce(val)
            extra_params.append(Parameter(name, ctype, cval))

        if is_new_node:
            node.set_parameters([Parameter(
                "pose_source", Parameter.Type.STRING, args.pose_source)])
            if extra_params:
                node.set_parameters(extra_params)
            node.replay_setup(args.behavior)
        elif args.veto_m is not None:
            node.set_parameters([Parameter(
                "new_landmark_proximity_veto_m",
                Parameter.Type.DOUBLE,
                float(args.veto_m))])

        # Lifecycle: LifecycleNode subclasses need on_configure +
        # on_activate driven explicitly in replay mode. The state
        # objects are unused by the callbacks, hence the simple stub.
        from rclpy.lifecycle import State as LifecycleState
        from lifecycle_msgs.msg import State as StateMsg
        _stub = LifecycleState(StateMsg.PRIMARY_STATE_UNCONFIGURED,
                               "unconfigured")
        node.on_configure(_stub)
        node.on_activate(_stub)

        residuals: list[dict] = []

        def _record_scan_legacy(t: float) -> None:
            if node._latest_result is None:
                return
            if node._gt_init_pose is None or node._latest_gt is None:
                return
            gt_now = _odom_to_pose3(node._latest_gt)
            gt_aligned = node._gt_init_pose.inverse().compose(gt_now)
            slam = node._latest_result.pose
            err = float(np.hypot(
                slam.x() - gt_aligned.x(),
                slam.y() - gt_aligned.y()))
            yaw_err = _yaw_err(
                slam.rotation().yaw(),
                gt_aligned.rotation().yaw())
            residuals.append({
                "t": t,
                "step": node._graph.step,
                "slam_x": slam.x(),
                "slam_y": slam.y(),
                "slam_yaw": slam.rotation().yaw(),
                "gt_x": gt_aligned.x(),
                "gt_y": gt_aligned.y(),
                "gt_yaw": gt_aligned.rotation().yaw(),
                "err_m": err,
                "yaw_err_rad": yaw_err,
            })

        def _record_scan_new(t: float) -> None:
            snap = node.replay_snapshot()
            if snap is None:
                return
            err = float(np.hypot(
                snap["slam_x"] - snap["gt_x"],
                snap["slam_y"] - snap["gt_y"]))
            yaw_err = _yaw_err(snap["slam_yaw"], snap["gt_yaw"])
            residuals.append({
                "t": t,
                "step": snap["step"],
                "slam_x": snap["slam_x"],
                "slam_y": snap["slam_y"],
                "slam_yaw": snap["slam_yaw"],
                "gt_x": snap["gt_x"],
                "gt_y": snap["gt_y"],
                "gt_yaw": snap["gt_yaw"],
                "err_m": err,
                "yaw_err_rad": yaw_err,
            })

        record_scan = _record_scan_new if is_new_node else _record_scan_legacy

        n_cones = 0
        n_other = 0

        # Read messages in chronological order. rosbag2's
        # SequentialReader interleaves topics by recording timestamp,
        # which is the right order for replay.
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            if topic not in required_topics:
                continue
            cls = topic_classes[topic]
            msg = deserialize_message(raw, cls)
            try:
                if is_new_node:
                    is_scan = node.replay_dispatch(topic, msg)
                else:
                    is_scan = False
                    if topic == "/imu":
                        node._on_imu(msg)
                    elif topic == "/motor_rpm":
                        node._on_rpm(msg)
                    elif topic == "/testing_only/odom":
                        node._on_gt_odom(msg)
                    elif topic == "/Conos_raw":
                        node._on_cones(msg)
                        is_scan = True

                if is_scan:
                    n_cones += 1
                    record_scan(t_ns * 1e-9)
                    if not args.quiet and n_cones % 50 == 0 and residuals:
                        last = residuals[-1]
                        print(f"  scan {n_cones} step={last['step']} "
                              f"err={last['err_m']:.2f}m "
                              f"yaw_err={np.degrees(last['yaw_err_rad']):+.1f}°")
                    if args.max_scans and n_cones >= args.max_scans:
                        break
                else:
                    n_other += 1
            except Exception as ex:
                print(f"\nSLAM crashed during {topic} (scan #{n_cones}): "
                      f"{type(ex).__name__}: {ex}", file=sys.stderr)
                return 1

        # ----- Summary -----
        print(f"\nReplay summary:")
        print(f"  cones: {n_cones} scans")
        print(f"  other: {n_other} samples (non-scan support topics)")

        if not residuals:
            print("  no residuals collected — calibration may not have "
                  "completed before the bag ended", file=sys.stderr)
            return 1

        errs = np.array([r["err_m"] for r in residuals])
        yaw_errs_deg = np.array(
            [abs(np.degrees(r["yaw_err_rad"])) for r in residuals])

        max_err_idx = int(errs.argmax())
        max_yaw_idx = int(yaw_errs_deg.argmax())

        print(f"  pose-vs-GT: "
              f"mean={errs.mean():.2f}m "
              f"max={errs.max():.2f}m "
              f"@scan={max_err_idx}/step={residuals[max_err_idx]['step']}")
        print(f"  yaw-vs-GT:  "
              f"mean={yaw_errs_deg.mean():.2f}° "
              f"max={yaw_errs_deg.max():.2f}° "
              f"@scan={max_yaw_idx}/step={residuals[max_yaw_idx]['step']}")

        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(residuals[0].keys()))
                w.writeheader()
                w.writerows(residuals)
            print(f"  residuals csv: {args.csv}")

        failed = False
        if errs.max() > args.max_pose_err_m:
            print(f"FAIL: max pose error {errs.max():.2f}m "
                  f"> threshold {args.max_pose_err_m:.2f}m", file=sys.stderr)
            failed = True
        if yaw_errs_deg.max() > args.max_yaw_err_deg:
            print(f"FAIL: max yaw error {yaw_errs_deg.max():.2f}° "
                  f"> threshold {args.max_yaw_err_deg:.2f}°", file=sys.stderr)
            failed = True
        return 1 if failed else 0
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
