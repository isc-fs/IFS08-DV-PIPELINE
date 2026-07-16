# Rosbag recording on the car

DV testing time is scarce, so we record a bag on **every** run — including
manual driving — for offline replay (perception / SLAM / odom development).

## TL;DR

| You run | Unit | Records | Bag prefix |
|---|---|---|---|
| `dv race` (or race boot) | `dv-pipeline` + `dv-record` | full graph incl. `/lidar_points`, once the pipeline is up | `race_` / `umbilical_` |
| `dv manual` | `dv-manual` | full **sensor** graph incl. `/lidar_points`, no autonomy | `manual_` |
| `dv stop` | — | stops both | — |

Bags land in `/home/isc/bags/`. **Always `dv stop` before copying a bag off the
car** — a hard kill leaves an unfinalized mcap (no footer/index).

## What gets recorded

`ros2 bag record -s mcap -a` — **every topic on the graph**, including the raw
`/lidar_points` cloud (~45 MB/s ≈ 2.7 GB/min; a 5-min run ≈ 13 GB). The cloud is
the one signal that lets us debug perception offline, so it is recorded by
default. Trim it with `DV_RECORD_EXCLUDE='^/lidar_points$'` if disk is tight.

**QoS matters:** the Hesai driver (and the uDV) publish best-effort. A default
(reliable) rosbag reader is QoS-incompatible with a best-effort publisher and
records **zero** messages, silently. The recorder forces a best-effort reader on
those topics (`DV_RECORD_BEST_EFFORT`); don't remove that or the bag comes back
empty for the cloud.

## When does it start?

Not on a mission — on the **graph being up**. The recorder waits for a warmup
topic to appear, then records:

- **race** (`dv-record`): waits for `/dv/status` (mission_control's heartbeat —
  the last node up). It is **pipeline-gated, not mission-gated**: it never reads
  `/ami/mission` and records the same way for every mission.
- **manual** (`dv-manual`): waits for `/lidar_points` to be advertised (the
  sensor layer is up). No pipeline, no `/dv/status`.

## Manual driving (no pipeline)

`dv manual` starts `dv-manual.service`, which brings up **only** the sensor layer
(`car_sensors.launch.py` — Hesai driver + static TFs) and records. It does **not**
start `mission_control` or the autonomy nodes.

```bash
dv manual         # start: sensors + recorder, bag = manual_<timestamp>
dv manual stop    # stop just the manual recorder
dv stop           # stop everything (pipeline + manual)
dv status         # shows dv-manual state + last bag
```

Notes:
- The uDV topics (`/imu`, `/motor_rpm`, `/steering_angle`) come from
  `microros-agent.service`, independent of this — they're in the manual bag iff
  that service is up (it normally is). The lidar cloud is started by
  `dv-manual` itself.
- **Don't run `dv manual` while the pipeline is up** — both launch the Hesai
  driver. `dv manual` refuses to start if `dv-pipeline` is active; `dv stop`
  first.
- Same disk floor (`DV_RECORD_MIN_FREE_GB`, default 20) and `/etc/dv/norecord`
  opt-out as the race recorder.

## Architecture (why it's split this way)

`car_bringup.launch.py` = `car_sensors.launch.py` + `car_pipeline.launch.py`.
The split lets `dv-manual` reuse the exact sensor layer without the autonomy
stack, and lets the manual and race recorders share one record core
(`dv_record.sh`) — manual just overrides the warmup topic/method and the bag
label via `DV_RECORD_*` env. The race path defaults are unchanged, so the split
is behaviour-neutral for `dv race`.

Files: `deploy/dv` (verbs), `deploy/dv-manual.service` + `deploy/dv_manual.sh`
(manual path), `deploy/dv-record.service` + `deploy/dv_record.sh` (shared core),
`bringup/launch/car_sensors.launch.py` (sensor layer).
