# Hard stop on mission completion (DV_STOPPING)

**Status: pipeline side implemented and INERT by default. The rules question is
ANSWERED and the firmware side is implemented (uDV#176, branch
`feat/176-dv-stopping`). Remaining gate: joint bench validation. Do not enable
on the car until that runs.**

## The problem

The pipeline cannot stop the car.

- The IFS08 has no mechanical service brake the pipeline can command.
- Regen is fully plumbed — `PIVelocity` computes it, `mission_control` relays it
  as a signed `/ctrl/cmd` (`linear.x = throttle - brake`) — but **nothing acts on
  it on the vehicle**. Regen braking has never been validated, so it is
  deliberately unimplemented on the uDV side.

So the car coasts. `v_max = 3.0 m/s` is what makes that tolerable today.

This breaks mission completion. `/slam/finished` is gated on standstill — and
correctly so, because entering AS Finished fires the EBS **and opens the SDC**,
so signalling it at speed would hard-stop the car mid-track. But with no braking
authority the car only coasts down, so standstill arrives late, imprecisely, or
not at all on any gradient. The **EBS is the only thing that can actually stop
the car**, which is why we want to use it deliberately rather than as a fault
response.

## The intent

On mission completion (accel, autocross, trackdrive, skidpad): actuate the
**pneumatic EBS components** to bring the car to a standstill, while

- **keeping the SDC closed**, and
- **staying in AS Driving** — this is *not* AS Emergency.

A "hard stop", not an emergency manoeuvre. Once at rest, the normal
`/slam/finished` → `DV_FINISHED` → AS Finished path runs as it does today.

## The question we asked, and the answer (uDV#176)

We asked: *the FS AS state table has no state with the EBS actuated and the SDC
closed — so is there an actuation path to the pneumatics that is NOT the EBS
trigger?*

**Answered, and the premise was a misreading.** That table describes **AS
states, not an electrical constraint**.

- **Independent of the SDC: yes.** The uDV drives the two EBS actuators
  (`D1`/`D2`) and the AS SDC (`D4`) as **three independent GPIOs**. The coupling
  between "fire the brakes" and "open the SDC" is a *policy* in
  `as_actuation.hpp` — not a wire.
- **"Brakes actuated + SDC closed" is not a new state at all.** It is already
  the **steady state of AS Ready**: the EBS is released only in AS Driving or
  with the ASMS off, and fired in every other state, while `sdc_open` is true
  only in Finished/Emergency. On top of that the T15.2 init self-check fires
  each actuator in turn, with the SDC closed, on **every boot**. The car is
  routinely braked-with-SDC-closed before it ever moves.
- **Independent of the EBS: no.** There is **no separate service brake**. The
  only brake pneumatics *are* the two EBS actuators, and they are **binary** —
  fire or release. Nothing to modulate, nothing to ramp.

### What that means for scope

Because the actuators are binary, this is **not** an autonomous service brake
and must not be used as one. It buys exactly one thing: **reaching standstill at
the very end of the mission so AS Finished becomes reachable.**

> ⚠️ **`DV_STOPPING` is full brake pressure, every time. Keep
> `hard_stop_on_finish` strictly end-of-mission. Never use it to modulate speed
> during a run** — every application is an emergency-grade stop.

`LapCounter.target_met` enforces this structurally: it only latches on the
mission's completion criterion, so there is no path that applies it mid-run.

The uDV team assessed that the narrow end-of-mission form does not need a rules
escalation — it is the EBS bringing the car to rest at mission end, which is
what it is for. They will still run it past scrutineering before a real run and
report back on uDV#176. If scrutineering pushes back, this stays gated off.

## The contract

New `/dv/status` byte:

```
DV_STOPPING = 7   # hard stop requested — brake to standstill, SDC stays closed
```

Sequence (trackdrive shown; every lap/distance mission is the same shape):

```
lap 9 crossed         → /slam/final_lap = true
                      → control_node arms its stop anchor for the closing gate

lap 10 crossed        → LapCounter.target_met latches (criterion met, still rolling)
                      → /slam/stop_request = true
                      → mission_control publishes /dv/status = DV_STOPPING
                      → uDV actuates the brake pneumatics
                           SDC stays CLOSED, AS stays DRIVING, no /force_ebs

car reaches standstill → LapCounter.finished (target_met AND stopped)
                      → /slam/finished = true
                      → mission_control publishes /dv/status = DV_FINISHED
                      → uDV runs the normal AS Finished actuation (EBS + SDC open)
```

### Why a `/dv/status` byte rather than a service

`/dv/status` is already the **actuating** path — per `docs/CAR_ADAPTATION.md`,
`DV_EMERGENCY` on this topic is what latches the uDV's EBS, while `/force_ebs`
is redundant and non-latching. Three consequences, all wanted:

- **It is a continuous 10 Hz heartbeat.** If the pipeline dies mid-stop,
  `/dv/status` goes stale past `DV_STATUS_STALE_MS` and the uDV trips to its
  safe state — EBS and SDC open. A one-shot service call would leave a dead
  pipeline and a rolling car. **Fail-safe by construction.**
- No new topic, no new QoS surface to get wrong.
- Consistent with the architecture: the pipeline states intent, the uDV owns
  actuation and the AS state machine. The pipeline never touches the AS machine.

### Byte priority

`DV_EMERGENCY` > `DV_FINISHED` > `DV_STOPPING` > `DV_FAILED`. A real fault
always outranks a tidy end-of-mission stop; once stopped, we are finished.

## Safety gating (pipeline side)

1. **`hard_stop_on_finish` parameter, default FALSE.** Note the reason has
   changed since this was written. Unknown-byte risk is **resolved**: the
   firmware compares `/dv/status` for equality against only the bytes it acts
   on, so byte 7 is inert on today's build (`RUNNING=3` has always been
   "unknown" to it in exactly the same way) and we can ship ahead of the
   firmware with no lockstep flash. The gate now exists solely because **the
   pairing has not been bench-validated**. With it off the pipeline behaves
   exactly as today: the car coasts and finishes on standstill.
2. **`ActiveLevel.RUNNING` only.** The free-run floor maps while a human drives;
   a stop request there would slam the brakes mid-manual-lap.
3. **Never while arming — `AS_DRIVING` only.** In AS Ready an unrecognised byte
   makes `dv_ready` false and the uDV **refuses GO**: the car silently never
   launches, with no obvious cause. `_current_dv_status` gates emission on the
   real AS state, not just on the latch — the latch clears only on full
   teardown, so a re-arm beating that reset would otherwise sit in AS Ready
   asserting 7.
4. **Latched.** `target_met` latches in the counter and `_stopping` latches in
   `mission_control`, so a flickering topic can never release a stop already
   under way.
5. **Never for a mission with no completion criterion.** Skidpad has none today,
   so it never requests a hard stop — see below.

### Firmware-side safety properties (uDV#176)

Worth knowing, because they shape what our side does *not* have to guard:

- **ASMS-off wins.** A marshal pushing the car gets released brakes even if a
  live pipeline is still asserting `STOPPING`. A stale 7 can never lock the
  wheels during recovery.
- **The byte can only ADD braking, never remove it** (asserted as a firmware
  invariant).
- **It never touches the SDC** — that is what keeps it a stop and not an
  emergency.
- **A stale link mid-stop does not release the brakes.** `dv_lost_driving`
  trips Emergency on the same tick, which fires the EBS anyway. The failure
  mode is "brakes stay on", not "car rolls".
- **Gated on `mission_needs_pipeline`**, so a stray 7 cannot brake a standalone
  mission.
- **Torque is zeroed** (`0x507`) so the car does not brake against its own
  drive. Steering is deliberately **not** inhibited — the wheels stay where you
  put them rather than snapping to centre mid-stop.

## Per-mission status

| Mission | Criterion | Hard stop |
|---|---|---|
| accel | 75 m distance | yes, once firmware lands |
| autocross | 1 lap | yes |
| trackdrive | 10 laps | yes |
| **skidpad** | **none** | **NO — see below** |

**Skidpad will not hard-stop**, despite being in scope for the request. It has
no completion criterion in `LapCounter` (its figure-8 breaks the
distance-from-origin lap model), so `target_met` never latches. Giving skidpad a
criterion is blocked on the separate skidpad loop-closure work — the same
geometry problem. Tracked separately; deliberately not bodged here.

## Testing

```bash
python -m pytest cone_slam/test/test_lap_counter.py mission_control/test/ -q
```

Run under real ROS in the `ros2-humble-dev` container: **113 passed, 0 skipped**.

Mutation-checked. Each of these fails tests: making `target_met` wait for
standstill (which would deadlock — never brake → never stop → never finish),
un-latching `target_met` (releases a stop mid-brake), and colliding
`DV_STOPPING` with `DV_EMERGENCY` (opens the SDC on a normal mission end).

**Not covered:** the `cone_slam` node cannot run in the container (no gtsam), so
its publish wiring is compile-checked only. And no test can prove the firmware
pairing.

## Joint bench validation (agreed on uDV#176)

The firmware ships `BENCH_STUB_DV_STOPPING` (default 0, so dev stays
flight-clean):

```bash
make BENCH="-DBENCH_STUB_DV_STOPPING=1"
```

With the stub, byte 7 is received, keeps the heartbeat fresh, and shows on
`/debug` and in the pit-diag stub mask (`0x40`) — but does **not** fire the EBS
and does **not** inhibit torque. So the whole handshake can be exercised while
the car still only coasts. Each real stop costs an air recharge, which is why
the first runs go this way.

Agreed order:

1. **Stub build** — emit `STOPPING`, confirm the handshake, `/debug`, and that
   no valve fires.
2. **Stub off, stationary** — confirm the valves fire and the SDC stays closed.
3. **Rolling** — confirm stop → `FINISHED` → AS Finished.

Only after (3) should `hard_stop_on_finish` be enabled for a real run.
