# Car adaptation — running the DV pipeline on the real vehicle

**Branch:** `feat/6-car-adaptation` (from `feat/5-sim-clock-imu-slam`)
**Status:** software adapters landed · firmware/inverter gaps flagged below
**Last verified against:** `IFS08-DV-uDV @ feat/14-bridge-parse-fixes`

This document is the source of truth for what was changed to make the
autonomy pipeline run on the car instead of the IFSSIM simulator, and —
critically — the **open gaps** that must be closed before an on-track
run. It supersedes the topic-name assumptions in
`08/DV_procedures/07_pipeline_car_adaptation.md` and
`00_dv_pipeline_roadmap.md`, which were written before the uDV firmware
contract was pinned (those docs assumed an `/isc/*` topic surface that
the firmware does **not** publish).

---

## 1. The real contract (verified against firmware, not assumed)

The car sensor/actuator surface is the **uDV micro-ROS node**
(`cubemx_node`, empty namespace) plus the **Hesai LiDAR driver** — not
the sim's `/fsds/*` bridge. Verified from `IFS08-DV-uDV` `freertos.c`
and `docs/PHYSICAL_TESTS.md`:

### uDV publishes
| Topic | Type | Notes |
|---|---|---|
| `/imu/data_raw` | `sensor_msgs/Imu` | 400 Hz |
| `/steering/angle_sensor` | `std_msgs/Float32` | **degrees**, ~10 Hz |
| `/assi/state` | `std_msgs/UInt8` | AS state 0/1/2/3/4 (T14.9) |
| `/ami/mission` | `std_msgs/Int32` | AMI mission index 0..9 |
| `/res/go`, `/res/status` | `std_msgs/Int32` | RES go-signal / status |
| `/steering/angle_actual`, `/steering/angle_target` | `std_msgs/Float32` | controller feedback |
| `/imu/status`, `/debug` | — | diagnostics |

### uDV subscribes / serves
| Name | Type | Notes |
|---|---|---|
| `/steering/cmd` | `std_msgs/Float32` | **degrees** → CAN `0x020` |
| `/assi/cmd` | `std_msgs/UInt8` | bench AS-state override |
| `/force_ebs` | `std_srvs/SetBool` (service) | EBS trigger |
| `/activate_steering` | `std_srvs/SetBool` (service) | steering motor enable |

### What the pipeline autonomy nodes consume (in source)
| In-code topic | Node(s) | Type / units |
|---|---|---|
| `/imu` | odometry_filter, slam | `sensor_msgs/Imu` |
| `/steering_angle` | odometry_filter, slam | `std_msgs/Float32`, **radians** |
| `/motor_rpm` | odometry_filter, slam | `std_msgs/Float32`, motor RPM |
| `/fsds/lidar/Lidar1` | cone_detection | `sensor_msgs/PointCloud2` |
| `/ctrl/cmd_internal` (out) | control → mission_control | `fs_msgs/ControlCommand` |

---

## 2. What was changed (this branch)

### a. Car remap table — `bringup/bringup/topic_contract.py`
A new dependency-free module holds the remap contract for both profiles
and a pure `autonomy_remaps(profile)` selector (unit-tested in
`bringup/test/test_topic_contract.py`, no ROS needed).
`autonomy_actions(profile="car")` and `car_pipeline.launch.py` use it.

Only **IMU** and **LiDAR** are pure remaps (type + units already match):

```
REMAP_IMU_CAR   = ("/imu",               "/imu/data_raw")
REMAP_LIDAR_CAR = ("/fsds/lidar/Lidar1", "/lidar_points")
```

> **Remap direction matters.** The first element is the topic the node
> uses *in its own source*. `cone_detection_node` hardcodes
> `/fsds/lidar/Lidar1`; `odometry_filter_node` hardcodes `/imu`. Getting
> this backwards silently subscribes to a topic nobody publishes — which
> is why the table is pinned by tests.

`/steering_angle` and `/motor_rpm` are **not** in the remap table: they
need conversion / have no direct source, so `car_sensor_bridge`
publishes them on their canonical names instead.

### b. `car_sensor_bridge` — input adapter
- `/steering/angle_sensor` (deg) → `/steering_angle` (rad). The EKF's
  `push_steering` is documented `angle_rad`; the uDV publishes degrees.
- inverter feed → `/motor_rpm` (motor-shaft RPM). The EKF applies its
  own `ekf.rpm_to_ms` scaling, so this topic must carry motor RPM.
- Pure conversions in `conversions.py` (unit-tested); node validates and
  drops non-finite samples.

### c. `car_supervisor` — mission/actuation adapter (replaces sim_supervisor)
- Subscribes `/assi/state` + `/ami/mission`; drives
  `mission_control_node` via `SetMission` (configure) → `RuntimeControl`
  (activate + run).
- Relays `RuntimeControl` feedback (throttle/steering in [-1,1]) to the
  uDV — **only while AS Driving** (`should_actuate`).
- Steering [-1,1] → degrees on `/steering/cmd`, hard-clamped to a safety
  limit kept under STEERING's 70° emergency cutoff.
- AS Emergency → `/force_ebs` service.
- Pure policy (`policy.py`) + scaling (`actuation.py`) are unit-tested.

---

## 3. ⚠️ OPEN GAPS — must close before on-track running

These are **not** pipeline-repo work; they are flagged here so they are
not missed. Each is parameterised in code so wiring is a one-line change.

### G1 — Inverter `/motor_rpm` source `[BLOCKER for odometry quality]`
The uDV exposes **no wheel-speed / motor-RPM topic**. The EKF needs
`/motor_rpm`; without it odometry is IMU-only dead-reckoning and drifts.
**The LattePanda has no SocketCAN interface**, so the inverter feed must
arrive as a ROS topic (relayed through the uDV, or via a future USB-CAN
bridge). Action items:
- Decide the transport (uDV CAN→ROS publisher, or Panda USB-CAN).
- Confirm the inverter reporting: **eRPM vs mechanical RPM**, units/LSB,
  sign, topic name, message type.
- Set `car_sensor_bridge` params: `inverter_in_topic`, `inverter_is_erpm`,
  `pole_pairs`, `inverter_scale`.
- Set `ekf.rpm_to_ms` on `odometry_filter_node` for the real wheel
  radius / gear ratio (the sim default `0.00821` is almost certainly wrong).

### G2 — Throttle / brake actuation sink
`RuntimeControl` feedback carries **throttle + steering only** (no
brake). The uDV currently has **no throttle/brake ROS subscriber** (only
`/steering/cmd`). `car_supervisor` publishes throttle on the placeholder
topic `/ctrl/throttle_cmd` so the command isn't lost. Action items:
- Add a uDV throttle subscriber (→ inverter torque/accel CAN, e.g. the
  `0x507` accel frame referenced in the uDV CLAUDE.md).
- Point `car_supervisor` `throttle_cmd_topic` at it.
- Proportional braking is out of the action contract; only emergency EBS
  is wired (`/force_ebs`).

### G3 — Steering scaling + units `[SAFETY]`
- `car_supervisor` `max_steering_deg` (default **20.0**) and
  `steering_safety_limit_deg` (default **25.0**) are PLACEHOLDERS.
  Measure the real full-lock command and set them (safety limit must
  stay under STEERING's 70° cutoff).
- `car_sensor_bridge` `steering_ratio` (default 1.0), `steering_sign`,
  `steering_offset_deg`: confirm whether `/steering/angle_sensor`
  measures road-wheel vs steering-column angle, and that the sign
  matches the EKF convention (positive δ ⇒ positive yaw rate).

### G4 — Mission-finished path to the uDV
The uDV has **no `/isc/mission_finished` (or equivalent) subscriber**.
On `slam/finished`, `RuntimeControl` closes and `car_supervisor` centres
the wheel, but nothing tells the uDV "mission complete" over ROS — the
uDV transitions DRIVING→FINISHED via its own RES/standstill logic.
Confirm this is acceptable, or add a uDV mission-finished subscriber.

### G5 — AMI → mission mapping
`car_supervisor` `policy.DEFAULT_AMI_TO_MISSION_ID` maps the AMI index
(uDV `ws2812.c`: 4=Track drive) to the pipeline registry
(trackdrive=1, autocross=2, accel=3, skidpad=4, scruti=5). Confirm
against AMI firmware. Two soft spots: AMI 5 "EVS/EBS test" and AMI 6
"Inspection" both map to `scruti`; and the uDV bench doc once said
"mission 5 = track drive" while the firmware table says index 4 — the
**firmware table is authoritative**.

### G6 — IMU / LiDAR frames + TF
`cone_detection` reads `header.frame_id`; the uDV stamps IMU with
`imu_link`, the Hesai cloud with its sensor frame. Provide the static
TFs (`base_link → hesai_lidar`, `base_link → imu_link`) — the values in
the DVPC `startup.launch.py` are currently placeholders. Re-tune the
RANSAC ground / DBSCAN clustering params on real ATX data.

---

## 4. Launching on the car

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/local_setup.bash   # RMW CycloneDDS + Hesai
source ~/dv_ws/install/setup.bash
ros2 launch bringup car_pipeline.launch.py    # NEVER with use_sim_time:=true
```

Per the repo workflow, only validated code on `main` runs on the car
(the `dv-pipeline-update` service fast-forwards `~/dv_ws` to
`origin/main`). This branch must reach `dev` → `main` before the
auto-update picks it up.

---

## 5. Tests

Pure-logic suites (no ROS install required):

```bash
python -m pytest bringup/test/test_topic_contract.py \
                 car_sensor_bridge/test/ \
                 car_supervisor/test/ -q
```
