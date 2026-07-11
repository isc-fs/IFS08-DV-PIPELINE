# `prerun/` — rosbag replay ON the car (pipeline side)

> ⚠️ **DO NOT DELETE this branch or this file.** `prerun/*` is a standing
> configuration branch, not a feature branch. It is never merged to `dev` and
> never deleted. If GitHub branch protection is set, deletion/force-push are
> blocked; keep it that way.

## What this branch is for

`prerun/` lets us **replay a recorded rosbag on the real car** (wheels off the
ground, on stands) to exercise the full autonomy + actuation stack **before**
putting the car on the ground. The IMU and LiDAR can't produce meaningful live
data in that state, so those two feeds come from the bag; **everything else
stays live** (EBS, steering, motor torque, RES, the DV handshake, ASSI).

This is the **pipeline half**. The firmware half lives on
`IFS08-DV-uDV @ prerun/rosbag-onboard` (which suppresses the live `/imu`
publish via `BENCH_STUB_IMU_ROS=1`).

## The changes vs `dev`

1. **`bringup/launch/car_bringup.launch.py`** — the `with_lidar` launch arg
   default is flipped **`true` → `false`**. `dv-pipeline.service` launches
   `car_bringup.launch.py` with defaults, so on this branch it boots **without
   the Hesai ATX driver**; the bag's `/lidar_points` is the only source (else
   the live driver and the bag collide). The static TFs
   (`base_link→hesai_lidar`, `→imu_link`) and all autonomy stay live. Pass
   `with_lidar:=true` to run the real driver.
2. **`deploy/prerun_restamp_relay.py`** — a small relay node that republishes
   the bagged `/imu` + `/lidar_points` with `header.stamp` rewritten to `now`
   (see below).

Nothing else diverges from `dev`. Keep it that way so rebasing stays trivial.

## Why re-stamp to "now"

The bag's sensor stamps are from a past recording, but the live DV handshake
(`dv/status` / `assi/state`) and the 400 ms staleness watchdogs run on current
time. Feeding old-stamped `/imu` + `/lidar_points` into a live-time system
would trip SLAM/odometry time logic and the watchdogs. The relay rewrites each
message's `header.stamp` to `now` as it republishes, so the live and replayed
clocks agree. (Static TFs are time-tolerant, so the live `base_link→hesai_lidar`
TF still applies to the re-stamped cloud.)

## Run procedure (DVPC, workspace sourced)

```bash
# 1) Pipeline up (this branch → Hesai driver off, autonomy consuming
#    /imu + /lidar_points). uDV flashed from its prerun/rosbag-onboard branch.
ros2 launch bringup car_bringup.launch.py        # or the dv-pipeline.service

# 2) Re-stamp relay (shadow topics -> real topics, stamped now):
python3 deploy/prerun_restamp_relay.py

# 3) Play the bag into the shadow topics, 1x so relative timing is preserved:
ros2 bag play <bag> --remap /imu:=/imu_bag /lidar_points:=/lidar_points_bag
```

## Rosbag requirements (shared with the uDV branch)

- The bag must contain **only** the replayed sensor feeds: `/imu`
  (sensor_msgs/Imu, `frame_id=imu_link`) and `/lidar_points`
  (sensor_msgs/PointCloud2, `frame_id=hesai_lidar`). If it also carries other
  uDV topics (`/motor_rpm`, `/steering_angle`, `/res/*`, `/assi/state`, …) they
  collide with the live firmware — strip them or add stubs.
- Do **not** include `/tf`/`/tf_static` for `hesai_lidar`/`imu_link` in the
  bag; those are published live by `car_bringup`.
- Play at **1x** so the re-stamped inter-sample timing matches real sensor dt.

## Verify

- After launch (before the bag): `ros2 node list` shows **no** Hesai driver;
  `ros2 topic info /lidar_points` → **0 publishers**.
- Relay up + bag playing: `ros2 topic info /lidar_points` and `/imu` → exactly
  **one** publisher each (the relay). More than one = collision / misconfig.
- `cone_detection` consumes `/lidar_points` → `/Conos_raw`; SLAM maps; control
  emits `/ctrl/cmd`; the uDV actuates — all with the car on stands.

## Keeping this branch fresh

Rebase on `dev` periodically so it stays current — the diff should collapse to
the `with_lidar` default flip, this file, and the relay script. **Never** merge
this branch into `dev`.
