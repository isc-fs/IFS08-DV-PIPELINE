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
dv status         # shows dv-manual + dv-automission state + last bag
```

### Automatic whenever the car is powered (zero-touch, no AMI)

You normally don't run `dv manual` by hand. `dv-automission.service` (enabled at
boot) watches the **uDV sensor stream** (`/imu`) and reconciles the manual
recorder to it — it keys off *"is the car on?"*, not any AMI selection:

| Condition | Automission does |
|---|---|
| car powered (`/imu` live), no pipeline | **starts** `dv-manual` → records |
| pipeline running (`dv race`) | **stops** `dv-manual` (pipeline's own recorder takes over) |
| `/imu` silent ≥ ~6 s (car off / uDV down) | **stops** `dv-manual` (finalizes the bag) |
| brief `/imu` blip | leaves current state (no thrash) |
| `/etc/dv/norecord` present | stops + stays idle |

So: **power the car → it records.** No AMI, no button, no commands to the car.
`dv race` hands off cleanly; `dv stop` (car still on) resumes manual recording.

**No firmware dependency** — `/imu` already publishes continuously with the ASMS
off (confirmed uDV#189). (We dropped the earlier `/ami/mission` trigger: the AMI
board only emits an index after the operator *confirms* a mission, so it needed
a button press — see uDV#189.)

> ⚠️ **Disk:** this records from power-on, **including idle setup time**, and the
> bag includes the ~45 MB/s lidar cloud — long powered-but-idle periods burn disk
> fast (~2.7 GB/min). `touch /etc/dv/norecord` to pause, `dv stop` when done, or
> `DV_RECORD_EXCLUDE='^/lidar_points$'` for a light telemetry-only bag. The
> `DV_RECORD_MIN_FREE_GB` floor still refuses to start below the free-space limit.

**Opt out:** `touch /etc/dv/norecord` (honoured live) or
`sudo systemctl disable --now dv-automission.service`.

### The full car sensor set

A manual bag captures **all four** on-vehicle sensors, from two sources:

| Topic | Type | Source | Started by |
|---|---|---|---|
| `/lidar_points` | PointCloud2 | Hesai ATX driver | `dv-manual` (car_sensors) |
| `/imu` | Imu | uDV | `microros-agent.service` |
| `/motor_rpm` | Float32 | uDV | `microros-agent.service` |
| `/steering_angle` | Float32 | uDV | `microros-agent.service` |

The lidar cloud alone is not enough for offline odom/SLAM replay — you need the
uDV's IMU + wheel-speed + steering too. Those come over `microros-agent.service`,
which `dv-manual` pulls in (`Wants=`), so it starts even during pure manual
driving with no pipeline.

`dv_manual.sh` **verifies the full set is live before recording** and logs
present/missing. If the uDV topics are absent (agent down / uDV link dead) it
records anyway (a cloud bag beats no bag) but **warns loudly** in the journal —
so a proprioceptive-less bag is never a silent surprise. It only refuses to
start if *nothing* is present.

> After `dv manual`, sanity-check: `dv status` (dv-manual active), then
> `ros2 bag info` on the `manual_<ts>` bag — confirm **non-zero message counts**
> for `/imu`, `/motor_rpm`, `/steering_angle` **and** `/lidar_points`, not just
> that the topics exist. The best-effort QoS override is what makes those counts
> non-zero; a wrong reader records the topic with zero messages.

### Other notes
- **Don't run `dv manual` while the pipeline is up** — both launch the Hesai
  driver. `dv manual` refuses to start if `dv-pipeline` is active; `dv stop`
  first.
- Same disk floor (`DV_RECORD_MIN_FREE_GB`, default 20) and `/etc/dv/norecord`
  opt-out as the race recorder.
- No GPS/GSS/brake sensor exists on the car — the four above are the complete
  proprioceptive + perception set the EKF and SLAM consume.

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
