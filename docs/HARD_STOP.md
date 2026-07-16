# Hard stop on mission completion (DV_STOPPING)

**Status: pipeline side implemented and INERT by default. Checked against
FS-Rules 2026 v1.1 and **compliant** (see below). Firmware side implemented
(uDV#176, branch `feat/176-dv-stopping`). Remaining gate: joint bench
validation. Do not enable on the car until that runs.**

> **Terminology, and it matters:** this is an **ASB actuation**, not an
> "EBS trigger". Those are different things in FS-Rules T14/T15, and the
> difference *is* the compliance argument. See "Why this is legal" below.

## The problem

The pipeline cannot stop the car.

- The IFS08 has no mechanical service brake the pipeline can command.
- Regen is fully plumbed — `PIVelocity` computes it, `mission_control` relays it
  as a signed `/ctrl/cmd` (`linear.x = throttle - brake`) — but **nothing acts on
  it on the vehicle**. Regen braking has never been validated, so it is
  deliberately unimplemented on the uDV side.

So the car coasts. `v_max = 3.0 m/s` is what makes that tolerable today.

This breaks mission completion. `/slam/finished` is gated on standstill — and
correctly so: entering AS Finished **opens the SDC**, which cuts the T15.2.2
supply path and thereby **activates the EBS** (T14.8.1) — so signalling it at
speed would hard-stop the car mid-track with the TS cut. But with no braking
authority the car only coasts down, so standstill arrives late, imprecisely, or
not at all on any gradient. The **ASB is the only thing that can actually stop
the car**, which is why we want to actuate it deliberately at mission end
rather than only ever as a fault response.

## The intent

On mission completion (accel, autocross, trackdrive, skidpad): perform an
**ASB brake actuation** to bring the car to a standstill, while

- **keeping the SDC closed** — load bearing, see below, and
- **staying in AS Driving** — this is *not* AS Emergency.

A "hard stop", not an emergency manoeuvre. Once at rest, the normal
`/slam/finished` → `DV_FINISHED` → AS Finished path runs as it does today —
and *that* step genuinely does activate the EBS.

The brake hardware is shared (T15.1.1: the ASB "features an EBS ... as part of
it"), which is exactly why the wording has to be precise: applying the brakes
is **not** the same as activating the EBS.

## Why this is legal (FS-Rules 2026 v1.1)

The worry was: *the AS state table has no state with the EBS actuated and the
SDC closed, so "brake but stay in AS Driving" must be illegal.* That is wrong,
for a precise reason.

**T14.8.1 defines the term narrowly.** The EBS is "activated" **only if the
T15.2.2 power supply path is cut** — not merely when the brakes are applied.
And T15.2.2 lists that path exhaustively:

> The EBS must be supplied by • **LVMS** • **ASMS** • the normally open contact
> of the relay according to T14.3.5 (RES bypass) • **a relay which is supplied
> by the SDC**

**The actuator lines are not in that list.** So with all four supply elements
intact — LVMS on, ASMS on, RES relay closed, and crucially **the SDC closed** —
asserting the actuators applies brake pressure while the EBS remains armed and
*not activated*. Figure 15 (T14.8.3) then takes its **left** branch, `R2D?` is
still true, and the car is legitimately **AS Driving** while braking.

**AS Ready is the existence proof.** T14.4.1 closes the SDC precisely *because*
"sufficient brake pressure is built up, i.e. brakes are closed". Brakes applied
+ SDC closed + EBS not activated is the state every FS car sits in before every
run. If actuating the brakes counted as EBS activation, no car could ever arm.

T14.4.1 also names the concept outright in its manual-driving clause — "the AS
has checked that **ASB is deactivated, i.e. no autonomous brake actuation
possible**". **Autonomous brake actuation** is a first-class rules concept,
distinct from EBS activation. That is what `DV_STOPPING` requests.

### The corollary: why `DV_FINISHED` is different

Figure 15's **right** branch — the only route to AS Finished — requires
`EBS activated? yes`. So the uDV opening the SDC at standstill is not
incidental: cutting the supply path is *what makes AS Finished reachable*.

> `DV_FINISHED` genuinely **does** activate the EBS.
> `DV_STOPPING` genuinely **does not**.
> Do not conflate them.

```
AS Driving, R2D, EBS armed but not activated
  → DV_STOPPING : ASB applies brakes; T15.2.2 supply intact; SDC CLOSED
  → still AS Driving ✓        (left branch — nothing exits R2D)
  → standstill
  → DV_FINISHED : SDC opens → supply path cut → EBS activated
  → "EBS activated? yes" → "mission finished & at standstill? yes"
  → "SDC open at RES?" NO     ← opened by the AS, not by the RES
  → AS Finished ✓
```

### What is NOT resolved by the rules

- **Scrutineering.** The uDV team will run this past scrutineering before a real
  run and report on uDV#176. If they push back, it stays gated off.
- **The schematic.** The argument holds provided the actuator lines sit
  *downstream* of the T15.2.2 supply chain rather than being series elements in
  it. T14.5.4 and the fact that AS Ready works both point that way, but the uDV
  team owns that detail.
- **T14.11.1** requires the ASF to document the whole AS *"including ASB"* —
  describe this there as an ASB actuation, not an EBS trigger.

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
- **It never touches the SDC** — which is exactly what keeps the EBS
  un-activated: the SDC relay is one of the four T15.2.2 supply elements, so
  leaving it closed keeps the supply path intact and the car in AS Driving.
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
`/debug` and in the pit-diag stub mask (`0x40`) — but does **not** actuate the
ASB (no valve fires) and does **not** inhibit torque. So the whole handshake can be exercised while
the car still only coasts. Each real stop costs an air recharge, which is why
the first runs go this way.

Agreed order:

1. **Stub build** — emit `STOPPING`, confirm the handshake, `/debug`, and that
   no valve fires.
2. **Stub off, stationary** — confirm the valves fire and the SDC stays closed.
3. **Rolling** — confirm stop → `FINISHED` → AS Finished.

Only after (3) should `hard_stop_on_finish` be enabled for a real run.
