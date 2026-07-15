# Hard stop on mission completion (DV_STOPPING)

**Status: pipeline side implemented, INERT by default. Blocked on uDV firmware
and on an unresolved rules question. Do not enable on the car yet.**

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

## ⚠️ The unresolved question — read this before implementing the firmware side

**The FS AS state table has no state with the EBS actuated and the SDC closed.**

| AS state | EBS | SDC |
|---|---|---|
| AS Off | unavailable → armed | open |
| AS Ready | **armed** | closed |
| AS Driving | **armed** | closed |
| AS Finished | **activated** | **open** |
| AS Emergency | **activated** | **open** |

In AS Driving the EBS is *armed* — charged and ready — not *activated*. Both
states that activate it also open the SDC. So "actuate the brakes but stay in
AS Driving" is outside the state table as written.

Whether that is legitimate reduces to one question for the mech/uDV side:

> **Is there an actuation path to the brake pneumatics that is NOT the EBS
> trigger?**

- **If yes** — the pneumatics can be commanded independently of the EBS
  fail-safe trigger — then this is simply the **autonomous service brake**.
  Normal, expected, and what the rules assume brings the car to rest before
  AS Finished. Implement it; nothing here is controversial.
- **If no** — the only path is the EBS trigger — then this asks the firmware to
  fire the EBS *without* its mandated SDC opening. That is a **rules and
  scrutineering question, not a code question**, and must be answered by a
  human against the specific ruleset before anyone wires it up.

This document does not answer that question. Tracked in `isc-fs/IFS08-DV-uDV`.

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

1. **`hard_stop_on_finish` parameter, default FALSE.** Current firmware has no
   case for byte 7 and its behaviour on an unknown `/dv/status` value is
   unverified, so the byte is never emitted until the team enables it. With it
   off the pipeline behaves exactly as today: the car coasts and finishes on
   standstill.
2. **`ActiveLevel.RUNNING` only.** The free-run floor maps while a human drives;
   a stop request there would slam the brakes mid-manual-lap.
3. **Latched.** `target_met` latches in the counter and `_stopping` latches in
   `mission_control`, so a flickering topic can never release a stop already
   under way.
4. **Never for a mission with no completion criterion.** Skidpad has none today,
   so it never requests a hard stop — see below.

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
side — that pairing needs a bench check once the uDV lands, starting with
"does the car actually stop, and does the SDC stay closed?"
