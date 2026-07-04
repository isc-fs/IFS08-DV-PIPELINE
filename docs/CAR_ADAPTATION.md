# Car adaptation — running the DV pipeline on the real vehicle

**Status:** stock-typed uDV ↔ mission_control interface landed · firmware
gaps flagged below
**Last verified against:** `IFS08-DV-uDV @ feat/14-bridge-parse-fixes`

This document is the source of truth for the interface between the
autonomy pipeline (the DVPC) and the car, and the **open gaps** that must
be closed before an on-track run.

## 0. The big picture — one pipeline, two "uDV"s

The autonomy stack (mode_manager, mission_control, odometry_filter,
cone_detection, slam, path_planning, control) runs **identical code in
sim and on the car**. The only thing that differs is *who plays the uDV*:

* **Real car:** the IFS-08 uDV (a micro-ROS endpoint) + the Hesai driver.
* **Sim:** `sim_supervisor` emulates the uDV; the IFSSIM bridge provides
  the sensors.

`mission_control` is the seam. It exchanges **only standard ROS 2
interface types** with the uDV, so the micro-ROS firmware needs **no
custom messages and no library recompile**, and the sim emulator can
speak the exact same surface. There are **no DVPC-side car adapter
nodes** — the old `car_sensor_bridge` and `car_supervisor` are gone;
their unit conversions, mission/AS logic and actuation scaling now live
in the uDV firmware (and, mirrored, in the sim emulator).

---

## 1. The stock-typed interface (uDV ↔ mission_control)

Single source of truth: `mission_control/interface_contract.py`. Every
type here is in the standard micro-ROS interface set.

### uDV → mission_control (uplink)
| Topic | Type | Notes |
|---|---|---|
| `/assi/state` | `std_msgs/UInt8` | AS state machine byte (FS-Rules T14.9). **Publish ≥2 Hz** — it is mission_control's liveness heartbeat. |
| `/ami/mission` | `std_msgs/Int32` | selected AMI mission **index** (0..9). |
| `/imu` | `sensor_msgs/Imu` | 400 Hz → consumed directly by autonomy `/imu` (no remap; canonical on both sides). |
| `/steering_angle` | `std_msgs/Float32` | **radians** — converted on-board from the deg sensor. |
| `/motor_rpm` | `std_msgs/Float32` | motor-shaft RPM from the inverter. |
| `/lidar_points` | `sensor_msgs/PointCloud2` | Hesai → autonomy `/fsds/lidar/Lidar1` (pure remap). |

### mission_control → uDV (downlink)
| Name | Type | Notes |
|---|---|---|
| `/dv/status` | `std_msgs/UInt8` | pipeline lifecycle byte (IDLE/PREPARING/READY/RUNNING/FINISHED/EMERGENCY/FAILED). The prepare/run **handshake** — the uDV gates "go" on `READY`. **Publish ≥2 Hz** (the uDV's liveness watchdog on the DVPC). |
| `/ctrl/cmd` | `geometry_msgs/Twist` | normalised command: `linear.x`=throttle, `angular.z`=steering, both [-1,1]. The uDV scales to physical units + clamps + actuates **only while AS Driving**. |
| `/force_ebs` | `std_srvs/SetBool` (service, **served by the uDV**) | mission_control requests EBS here on emergency. |

The two `UInt8` byte topics are each other's heartbeats: a stale
`/assi/state` makes mission_control reconcile to torn-down; a stale
`/dv/status` holds the uDV in a safe state. This bidirectional liveness
replaces the old action goal's implicit connection.

### How a mission runs (the handshake)
1. AMI selects a mission → uDV publishes `/ami/mission`.
2. Operator arms → uDV asserts **AS Ready** → mission_control configures
   the autonomy → publishes `/dv/status = PREPARING → READY`.
3. **The uDV gates the RES go on `/dv/status == READY`** — it won't enter
   AS Driving until autonomy is genuinely prepared.
4. RES go → uDV → **AS Driving** → mission_control activates → publishes
   `/dv/status = RUNNING` and streams `/ctrl/cmd`.
5. `slam` finished → mission_control `/dv/status = FINISHED`; emergency →
   `/dv/status = EMERGENCY` + `/force_ebs` call.

mission_control is a **reconciler**: it reads `/assi/state` +
`/ami/mission` level-triggered and drives `mode_manager` so the autonomy
lifecycle converges to what the AS state demands. Decision logic is the
pure, unit-tested `mission_control/reconcile.py`.

---

## 2. What replaced car_sensor_bridge / car_supervisor

| Old DVPC adapter responsibility | Now lives in |
|---|---|
| steering deg→rad | uDV firmware (publishes `/steering_angle` in rad) |
| inverter→`/motor_rpm` | uDV firmware (reads inverter CAN) |
| AS-state → mission/actuation | uDV firmware AS state machine + `/ctrl/cmd` subscriber |
| `[-1,1]`→deg steering scaling + clamp | uDV firmware |
| "actuate only while Driving" gate | uDV firmware (it owns AS state) |
| AMI index→registry mission_id | mission_control (`interface_contract`, Python, testable) |
| EBS request | mission_control calls the uDV's `/force_ebs` |

In sim, `sim_supervisor` performs the firmware half in Python (AS state
machine in `sim_supervisor/as_state_machine.py`, `/ctrl/cmd`→bridge
relay, `/force_ebs` server), driven by the backend/CLI over the sim
operator-panel topics (`/sim/mission`, `/sim/intent`, `/sim/estop`).

---

## 3. ⚠️ OPEN GAPS — must close before on-track running

These are **firmware** items (IFS08-DV-uDV), flagged here so they aren't
missed. Each is a bounded change.

### G1 — Inverter `/motor_rpm` source `[BLOCKER for odometry quality]`
The uDV exposes **no wheel-speed / motor-RPM topic** yet. The EKF needs
`/motor_rpm`; without it odometry is IMU+steering dead-reckoning and
drifts. The uDV is on the vehicle CAN bus, so it is the natural publisher.
Action: read the inverter CAN, confirm **eRPM vs mechanical RPM** /
units / sign / pole-pairs, publish `/motor_rpm` (motor-shaft RPM). Set
`ekf.rpm_to_ms` on `odometry_filter_node` for the real wheel radius /
gear ratio (the sim default `0.00821` is almost certainly wrong).

### G2 — Throttle actuation sink
`/ctrl/cmd` carries throttle (`linear.x`) + steering (`angular.z`); the
uDV currently has **no throttle ROS subscriber** (only steering). Add the
inverter torque/accel path (e.g. the `0x507` accel frame). Proportional
braking is out of scope; only emergency EBS is wired (`/force_ebs`).

### G3 — Steering scaling + units `[SAFETY]`
`/ctrl/cmd.angular.z` is normalised [-1,1]; the uDV scales it to degrees,
clamps to a safety limit **under STEERING's 70° cutoff**, and applies
only while AS Driving. Measure the real full-lock command and the sign
(positive δ ⇒ positive yaw rate) and bake them into firmware. Likewise
confirm the `/steering_angle` deg→rad conversion measures road-wheel vs
column angle.

### G4 — Mission-finished path to the uDV
On `slam/finished`, mission_control publishes `/dv/status = FINISHED`.
The uDV should react (DRIVING→FINISHED) — confirm it does, either via
`/dv/status` or its own RES/standstill logic.

### G5 — AMI → mission mapping
`interface_contract.DEFAULT_AMI_TO_MISSION_ID` maps the AMI index (uDV
`ws2812.c`: 4=Track drive) to the registry (trackdrive=1, autocross=2,
accel=3, skidpad=4, scruti=5). Confirm against AMI firmware. Soft spots:
AMI 5 "EVS/EBS test" + AMI 6 "Inspection" both map to `scruti`; the
firmware table (index 4 = Track drive) is authoritative.

### G6 — IMU / LiDAR frames + TF
`cone_detection` reads `header.frame_id`; provide the static TFs
(`base_link → hesai_lidar`, `base_link → imu_link`) and re-tune the
RANSAC ground / DBSCAN params on real ATX data.

### G7 — micro-ROS entity budget
The new uDV-facing topics (`/ctrl/cmd` sub, `/dv/status` sub,
`/motor_rpm` + `/steering_angle` pubs) add entities. Confirm they fit the
firmware's `RMW_UXRCE_MAX_*` budget; if not, bump `colcon.meta` and
rebuild (a config rebuild, **not** a custom-type integration).

---

## 4. Launching

```bash
# Car (NEVER use_sim_time:=true; the uDV is the mission_control peer):
ros2 launch bringup car_pipeline.launch.py

# Sim (sim_supervisor is the uDV emulator; backend/CLI drive the panel):
ros2 launch bringup full_pipeline.launch.py use_sim_time:=true
ros2 run sim_supervisor supervisor_cli run 1     # e.g. trackdrive
```

Per the repo workflow, only validated code on `main` runs on the car.

---

## 5. Tests

Pure-logic suites (no ROS install required):

```bash
python -m pytest pipeline/bringup/test/test_topic_contract.py \
                 pipeline/mission_control/test/ \
                 ros2/src/sim_supervisor/test/test_as_state_machine.py -q
```

Covers: the remap table, the AS/DV byte contract + AMI map, the
mission_control reconciler decision table, and the sim emulator AS state
machine (which shares the canonical bytes with `interface_contract`, so
the test also guards that the two packages agree).
