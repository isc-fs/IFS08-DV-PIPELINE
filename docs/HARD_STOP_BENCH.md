# DV_STOPPING bench validation

The three-stage bench test that must pass **before** `hard_stop_on_finish` is
enabled on a real run. Agreed with the uDV team on isc-fs/IFS08-DV-uDV#176.

`DV_STOPPING` fires **full ASB brake pressure** (the actuators are binary — no
modulation). So we validate the signal path first, then the actuation stationary,
then the whole flow rolling — spending an air recharge only when we have to.

## Preconditions — check ALL before starting

- [ ] uDV firmware with byte 7 (`DV_STATUS_STOPPING = 7u`) **flashed to the car.**
      Merged to uDV `dev` is not enough — confirm the running build. On a stock
      firmware (bytes 0–6) byte 7 is inert, so a stale flash silently no-ops the
      whole test.
- [ ] Car on stands / brakes-on-stands rig, wheels free, TS available per your
      bench SOP. ASR present (A4.4).
- [ ] ASMS reachable and its lockout understood — **ASMS-off releases the ASB**
      even mid-STOPPING (uDV#176), so it is your hard abort.
- [ ] Pipeline built from a branch containing the launch arg (this branch or
      later): `hard_stop_on_finish` defaults **false**; you enable it per-run.

## What to watch (three terminals)

```bash
# 1. The downlink byte — this is the whole test. 7 = STOPPING, 4 = FINISHED.
ros2 topic echo /dv/status std_msgs/msg/UInt8

# 2. The trigger from SLAM (or your manual inject).
ros2 topic echo /slam/stop_request std_msgs/msg/Bool

# 3. mission_control's own words — it logs exactly what it decided.
#    "requesting DV_STOPPING ..."  = it emitted byte 7
#    "... hard_stop_on_finish is off ..." = flag not set (expected until you flip it)
ros2 launch bringup car_pipeline.launch.py free_run:=false   # + the stage flag below
```

On the firmware side, watch `/debug` and the pit-diag stub mask (`0x40` = the
STOPPING stub is seeing the byte) per the uDV bench notes.

---

## Stage 1 — signal path, STUB firmware, no valve fires

**Goal:** prove the whole handshake end-to-end while the car still only coasts.

Firmware: build with the stub so byte 7 is received and shown but does **not**
actuate:

```bash
make BENCH="-DBENCH_STUB_DV_STOPPING=1"     # uDV side
```

Pipeline: enable the flag for this run.

```bash
ros2 launch bringup car_pipeline.launch.py free_run:=false hard_stop_on_finish:=true
```

Drive/emulate to a live mission so the AS is in **Driving** (the byte is only
emitted while `ActiveLevel.RUNNING` **and** the real AS state is `AS_DRIVING` —
see `_current_dv_status`). Then trigger a stop request **without** needing SLAM
to detect a real finish:

```bash
ros2 topic pub --once /slam/stop_request std_msgs/msg/Bool "{data: true}"
```

**Pass criteria**
- [ ] mission_control logs `... requesting DV_STOPPING ...`
- [ ] `/dv/status` goes to **7**
- [ ] uDV `/debug` shows byte 7 received; pit-diag mask shows `0x40`
- [ ] **No valve fires** — the car does not brake (stub build)
- [ ] AS stays **Driving** (ASSI yellow flashing), **not** Emergency (blue + sound)

If `/dv/status` never reaches 7: check the flag actually took
(`ros2 param get /mission_control_node hard_stop_on_finish` → should be `true`),
and that you were in AS Driving, not AS Ready — the AS_DRIVING gate suppresses
the byte during arming on purpose (an unknown byte in AS Ready refuses GO).

---

## Stage 2 — actuation, REAL firmware, STATIONARY

**Goal:** the valves actually fire, and the SDC stays closed. This is the
compliance property (ASB actuation, EBS armed-not-activated, AS still Driving).

Firmware: normal build (stub **off**). Car **stationary** on the rig, TS up.

Same launch + same manual trigger as Stage 1:

```bash
ros2 launch bringup car_pipeline.launch.py free_run:=false hard_stop_on_finish:=true
# ... reach AS Driving, then:
ros2 topic pub --once /slam/stop_request std_msgs/msg/Bool "{data: true}"
```

**Pass criteria**
- [ ] `/dv/status` = **7**
- [ ] Brake **valves fire** — measurable ASB pressure at the calipers
- [ ] **SDC stays CLOSED** (TS stays up; AIRs stay closed) ← the key compliance check
- [ ] AS stays **Driving**, **not** Emergency
- [ ] Torque command is zeroed (`0x507`) — no drive fighting the brake
- [ ] ASMS-off **releases** the brakes even while byte 7 is still asserted (abort path)

Each fire costs an air recharge — plan the reps.

---

## Stage 3 — full flow, REAL firmware, ROLLING

**Goal:** the real mission-end sequence, moving.

Firmware: normal build. Run a **short real mission** (or a controlled roll on the
rig) so SLAM's `LapCounter` reaches `target_met` on its own and publishes
`/slam/stop_request` — no manual inject this time.

```bash
ros2 launch bringup car_pipeline.launch.py free_run:=false hard_stop_on_finish:=true
```

**Pass criteria — the whole chain**
- [ ] At the mission criterion, `/slam/stop_request` rises on its own
- [ ] `/dv/status` = **7**, car brakes to a **standstill** (AS still Driving)
- [ ] At standstill, `/slam/finished` rises → `/dv/status` = **4**
- [ ] uDV transitions **AS Driving → AS Finished** (ASSI blue continuous, SDC opens)
- [ ] Deceleration is stable (T15.4.4); the car does not slew

---

## After the test

- **Only if all three stages pass** is `hard_stop_on_finish:=true` appropriate for
  a real competition run — and it is set **per-run at launch**, never as a source
  default. The launch default stays `false` (pinned by
  `test_hard_stop_launch_default.py`).
- Record the run: it feeds the ASF (T14.11.1 — document the AS "including ASB").
- If any stage fails, `hard_stop_on_finish` stays **off**; the car reverts to
  coasting to a stop and finishing on standstill, exactly as before this feature.

## Abort at any point

- **ASMS off** → ASB releases (uDV#176). Primary abort.
- **RES e-stop** → opens the SDC → EBS activated → AS Emergency. Full stop.
- Ctrl-C the launch → `/dv/status` goes stale → uDV trips to its safe state after
  `DV_STATUS_STALE_MS` (400 ms). The pipeline dying mid-stop is fail-safe by
  construction (that is why `DV_STOPPING` is a heartbeat byte, not a one-shot).
