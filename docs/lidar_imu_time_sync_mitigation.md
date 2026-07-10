# LiDAR ↔ IMU time-sync: analysis & mitigation plan

**Status:** proposal (nothing implemented). **Date:** 2026-07-07.
**Scope:** DVPC (LattePanda) + `cone_slam` on `main`. **Audience:** whoever picks up the SLAM timing work.

---

## TL;DR

- The LiDAR clouds are stamped with **DVPC system time** (`use_timestamp_type: 1`, receive-time), the same clock the IMU/EKF use — so there is **no gross clock mismatch**.
- The "~123 ms" figure is **LiDAR data *delivery* latency** (how old a cloud is when the SLAM processes it), **not a timestamp bias**. Do **not** "fix" it by subtracting 123 ms from a stamp — that would be wrong.
- Where it actually hurts: in the **default `motion_model="odom"`** path, cone observations (measured at the scan time `t_scan`) are attached to the **latest** EKF `/odom` pose, which is ~one delivery-latency *later*. Under acceleration/turning this smears the cone map by `latency × velocity` (≈ 1.2 m at 10 m/s).
- **Primary fix:** time-align the odom pose lookup to `t_scan` (interpolate a buffered `/odom`, don't use "latest"). Secondary: cut the cloud size to reduce the latency. Prerequisite: verify the uDV IMU is micro-ROS time-synced when the board is connected.

---

## 1. Background — two sensors, one clock (now)

SLAM fuses two sensor streams:

| Stream | Source | Transport | Clock |
|---|---|---|---|
| `/lidar_points` (→ cones) | Hesai ATX LiDAR | Ethernet (`enp1s0`) | DVPC **system time** (receive-time stamping) |
| `/imu` → `/odom` (EKF) | uDV firmware | micro-ROS (serial `/dev/ttyMicroDV`) | DVPC system time **iff** micro-ROS time-synced |

The LiDAR was **not** on system time until recently: it free-runs on its own clock and does **not** slave to PTP (its clock source is not set to PTP; confirmed via a read-only PTC `GetLidarStatus` → `ptp_status = 0 / free run`, under both L2 and UDPv4 PTP masters). Fixing that needs Hesai's config tool on a laptop — see Appendix A. As a working substitute we use **`use_timestamp_type: 1`** so the driver stamps each cloud with the DVPC system clock at receive. That puts LiDAR and IMU on the **same clock**, which is what SLAM needs.

---

## 2. What the "123 ms" is — and is not

Measured: for a freshly received cloud, `system_now − cloud.header.stamp ≈ 0.12 s`.

This is the **age of the message when a subscriber gets it**: driver stamps at end-of-frame reception → serializes a 174 000-point / **4.5 MB** cloud → best-effort DDS delivery → subscriber callback. It is **delivery/processing latency**.

It is **not** a timestamp error: the `header.stamp` still correctly refers to (end-of-)scan time on the system clock. So:

- ✅ Time-based association that uses the **stamp** is fundamentally sound.
- ❌ "Subtract 123 ms from the LiDAR stamp" is the wrong mental model — the stamp isn't 123 ms early/late; the *data arrives* 123 ms after it was stamped.

There is a smaller, separate modelling bias — the stamp is ~end-of-scan while the 100 ms of points span the whole rotation — addressed by deskew (option D), not by an offset.

---

## 3. Where the latency actually bites (code walk-through)

`cone_slam/cone_slam/cone_graph_slam_node.py` runs one of two motion models:

### 3a. `motion_model = "imu"` (legacy, OFF by default)
`_on_cones` → `self._preint.integrate_to(t_scan)` (~line 1197) integrates buffered IMU up to the LiDAR `t_scan`, builds an `ImuFactor`. This path **is** timestamp-sensitive, but it is not the default (`declare_parameter("motion_model", "odom")`, ~line 247) and the code comments call the EKF path "immune to the degenerate-timestamp preintegration error." **Not the active concern.**

### 3b. `motion_model = "odom"` (DEFAULT) — the real issue
In `_on_cones`, the per-scan motion is the delta of the EKF `/odom`:

```python
cur_odom_pose = _odom_to_pose3(self._latest_supervisor_odom)   # LATEST odom, not odom@t_scan
between_pose  = self._prev_scan_odom_pose.inverse().compose(cur_odom_pose)
predicted_pose = self._graph.stage_odom_motion_step(between_pose, ...)
```

`_latest_supervisor_odom` is set by `_on_supervisor_odom` (~line 1752) as simply *"cache the latest sample"* — **no time-matching to the scan**. Cone BearingRange factors are then attached at that pose.

**Consequence:** the cones were measured at `t_scan`, but they are attached to the pose at *"latest odom when the cone message arrived"* ≈ `t_scan + delivery_latency`. With **constant** latency the *relative* scan-to-scan motion mostly cancels, but during **acceleration or turning** the pose has moved a different amount in that window, so each cone lands `latency × velocity` off in a direction that changes with the maneuver → **cone-map smearing exactly during dynamic driving**, which is where cone SLAM must be sharpest.

---

## 4. Mitigation options (ranked)

### A. Time-align the odom lookup to `t_scan` — **primary fix**
Replace "use latest odom" with "use the odom pose **interpolated to `t_scan`**".

- Turn `_on_supervisor_odom` into a small **time-ordered ring buffer** of recent `/odom` samples (a few hundred ms is plenty).
- In the `odom` motion branch, compute `cur_odom_pose = pose_at(t_scan)` by interpolating between the two buffered samples bracketing `t_scan` (SLERP yaw, lerp position), instead of `self._latest_supervisor_odom`.
- Do the same for `_prev_scan_odom_pose` (it's already "odom at the previous scan's processing time" — make it "odom at the previous `t_scan`" for consistency).

**Why it's correct:** cones attach to the pose at their true measurement time; robust to both the mean latency *and its jitter*. Localized to `cone_graph_slam_node`.
**Cost/risk:** touches the pose used for every cone factor. Must (1) fall back to "latest odom" when `t_scan` is outside the buffer (startup, `/odom` dropouts), (2) lock the buffer — cone and IMU callbacks run under a `MultiThreadedExecutor` (mirror the existing `_preint` reentrant lock), (3) ship behind a parameter (e.g. `odom_time_align`) so it can be toggled and A/B-compared.
**Alternative mechanism:** a `tf2_ros.Buffer` lookup of `odom → base_link` at `t_scan` (if the EKF publishes that TF with correct stamps) achieves the same without hand-rolling interpolation.

### B. Cut the delivery latency at the source
The 123 ms is dominated by shipping 4.5 MB. Reduce the cloud:
- **Azimuth FOV crop** — already tooled on the DVPC: `fov <start> <end>` (see `deploy`/on-box `dv-lidar-fov`; edits the Hesai `config.yaml` `fov_start`/`fov_end` and restarts). 174 k points is far more than cone detection needs.
- Optional **voxel/rate downsample** before publish.
Smaller clouds → lower and less-jittery latency, and lower control lag. Complements A; not a substitute for it.

### C. Verify uDV IMU micro-ROS time-sync — **prerequisite, not optional**
Everything above assumes `/imu` (and therefore the EKF `/odom`) is on DVPC system time. That holds only if the uDV is micro-ROS **time-synced** to the agent. If it publishes on its own clock, `/odom` timestamps are wrong and A cannot help. **Check the instant the board is on `/dev/ttyMicroDV`:** compare `/imu` `header.stamp` to `date +%s` on the DVPC; they must match within milliseconds.

### D. Per-point deskew (separate, larger scope)
Even with a perfect frame stamp, a cloud spans ~100 ms of rotation; a moving car needs per-point-time **deskew** against the IMU to undo intra-scan distortion. Requires the cloud's per-point timestamp field (unconfirmed — verify the `PointCloud2` fields) and a deskew stage before cone detection. Only pursue if A+B leave residual smear at speed.

### E. Root fix — PTP measurement-time (needs Hesai tool)
Set the LiDAR clock source to PTP so clouds carry true measurement-time stamps (and enable clean deskew). The DVPC PTP master is **already fixed and waiting** (see Appendix A); this is blocked only on the LiDAR-side config, which needs Hesai's utility on a laptop. Separate track.

---

## 5. Recommended sequence

1. **C** — verify IMU time-sync when the uDV is connected (gates everything).
2. **A** — time-aligned odom lookup, behind a parameter. Highest value, correctly scoped.
3. **B** — FOV crop / downsample (nearly free with the existing `fov` tool).
4. Re-measure; only then consider **D**, and **E** as a longer-term hardware-config track.

**Validation metric:** with the uDV connected, log `t_scan` vs the `/odom` stamp actually used, and measure **landmark spread on re-observation during a hard turn** (a stable cone should not walk). A should visibly tighten it; B should reduce its variance.

---

## Appendix A — PTP / LiDAR clock state (for context)

> **RESOLVED 2026-07-10 — it was a PTP *profile* mismatch, not a clock-source problem.** The earlier "LiDAR clock source is not set to PTP" conclusion was wrong. On-wire capture proved the ATX runs **802.1AS / gPTP** (P2P peer-delay, `Pdelay_Req`, `transportSpecific=1`, multicast `01:80:C2:00:00:0E`), while our `ptp4l` ran **default 1588v2 E2E** (`transportSpecific=0`, multicast `01:1B:19:00:00:00`). gPTP discards every `transportSpecific=0` frame, so the two were mutually invisible → `ptp_status=0`. Fix = switch the **master** to gPTP; **no LiDAR-side change, no PandarView needed.**

- **DVPC PTP master: now gPTP.** `/etc/linuxptp/ptp4l.conf` set to `delay_mechanism P2P`, `transportSpecific 0x1`, `ptp_dst_mac 01:80:C2:00:00:0E`, `follow_up_info 1`, `path_trace_enabled 1`, `network_transport L2`, `logSyncInterval -3`, kept `time_stamping software` (serves `CLOCK_REALTIME`). Versioned + reproducible at `deploy/ptp4l.conf` (+ `deploy/ptp4l.service`, `deploy/install_ptp4l.sh`). `pmc` confirms `portState MASTER`, `delayMechanism 2 (P2P)`, measured `peerMeanPathDelay`. Rollback: `~/.isc_backups/ptp4l.conf.pre-gptp`.
- **LiDAR: now slaving to PTP.** After the switch, `GetLidarStatus` (PTC 1.0, port 9347) reports `ptp_status = 1` (**Tracking**, was `0`/free-run) and `GetPTPDiagnostics` qt=1 returns a live, bounded offset (≈ ±100–300 µs, oscillating around 0) with ~400–670 ns path delay. Bounded-around-zero = disciplined to the master; the ±300 µs residual is the **software-timestamping** regime (won't reach the ±1 µs "Locked" state). NB: the ATX `GetLidarStatus` payload is 58 bytes / 9 temperatures — `ptp_status` is at **byte offset 52**, not the XT 49-byte offset.
- **NIC note:** `enp1s0`/`enp2s0` are **Intel i226 (`igc`)**, not Realtek — both have working PHCs, so **hardware timestamping is available**. For tighter LiDAR lock later, move the master to `time_stamping hardware` **and add `phc2sys -s enp1s0 -c CLOCK_REALTIME`** (else the LiDAR follows the NIC PHC, not the system clock, and drifts from the IMU).
- **Driver timestamp:** `use_timestamp_type: 1` (receive-time) is still in `config.yaml`. Now that the LiDAR clock tracks system time, switching to `use_timestamp_type: 0` (point-cloud/measurement time) is the recommended follow-up once lock quality is confirmed acceptable in the pipeline.

## Appendix B — key code references (`main`)

- `cone_slam/cone_slam/cone_graph_slam_node.py`
  - `declare_parameter("motion_model", "odom")` — default motion model (~L247)
  - `_on_cones` — odom motion branch uses `_latest_supervisor_odom` (~L1106+); legacy imu branch `integrate_to(t_scan)` (~L1197)
  - `_on_supervisor_odom` — caches latest `/odom`, no time-match (~L1752)
  - `_on_imu` — buffers IMU on its own `header.stamp` (~L1017)
- `cone_slam/cone_slam/imu_preintegrator.py` — `integrate_to(t_end)` semantics (buffer reset on each call)
- LiDAR FOV tool (DVPC): `fov <start> <end>` → `/etc/dv/lidar_fov.yaml` + Hesai `config.yaml`
