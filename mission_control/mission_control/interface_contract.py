"""Stock-typed uDV ↔ mission_control interface — the single source of truth.

The DV pipeline and the "uDV" (the real micro-ROS endpoint on the car, or
sim_supervisor emulating it in the sim) exchange ONLY standard ROS 2
interface types, so the firmware needs no custom messages and no
micro-ROS library recompile. mission_control runs one identical reconciler
against this surface in both worlds; the sim/car difference is only *who*
plays the uDV.

  uDV → mission_control (uplink, the "request"):
    /assi/state   std_msgs/UInt8   AS state machine byte (FS-Rules T14.9)
    /ami/mission  std_msgs/Int32   selected AMI mission INDEX (0..9, raw)

  mission_control → uDV (downlink, the "ack" + command):
    /dv/status    std_msgs/UInt8   pipeline lifecycle byte — the stock
                                   stand-in for the old SetMission /
                                   RuntimeControl action Results
    /ctrl/cmd     geometry_msgs/Twist  normalised command:
                                   linear.x = throttle [-1, 1],
                                   angular.z = steering [-1, 1]
    /finish_brake std_msgs/Bool   finish service-brake request: the uDV
                                   engages the EBS actuators WITHOUT opening the
                                   SDC (heavy controlled stop, stays AS Driving
                                   until standstill). A latched topic, not a
                                   service (the uDV's service slots are full).
    /force_ebs    std_srvs/SetBool (service)  emergency-brake request

Both byte topics MUST be published at a steady cadence (>= ~2 Hz, not
edge-only): each is the other side's liveness heartbeat (a stale
/assi/state deactivates the pipeline; a stale /dv/status holds the uDV in
a safe state).

This module is pure data + pure functions — NO rclpy / launch imports —
so it unit-tests in plain pytest and can be imported by both
mission_control (the reconciler) and sim_supervisor (the emulator). The
same byte values are mirrored in the uDV firmware (C); this module is the
canonical Python source.
"""
from __future__ import annotations


# AS state bytes on /assi/state (FS-Rules T14.9 / uDV MMEE). The uDV —
# never the DVPC — owns this state machine; mission_control only reacts.
AS_OFF       = 0
AS_EMERGENCY = 1
AS_READY     = 2
AS_DRIVING   = 3
AS_FINISHED  = 4

# /dv/status bytes — the pipeline's own lifecycle, reported back to the
# uDV as the prepare/run handshake. The uDV gates AS Ready on DV_READY
# (won't honour "go" until autonomy is genuinely prepared) and reacts to
# DV_FINISHED / DV_EMERGENCY / DV_FAILED.
DV_IDLE      = 0   # nothing prepared
DV_PREPARING = 1   # configure / JIT in flight (was SetMission feedback)
DV_READY     = 2   # prepared OK (was SetMission Result success=true)
DV_RUNNING   = 3   # activated, emitting /ctrl/cmd
DV_FINISHED  = 4   # mission complete (was RuntimeControl outcome=finished)
DV_EMERGENCY = 5   # pipeline raised EBS (was outcome=emergency)
DV_FAILED    = 6   # prepare/activate error (was Result success=false/error)

# Interface topic / service names — single source of truth so the
# reconciler, the emulator and any tooling never drift.
TOPIC_ASSI_STATE   = "/assi/state"
TOPIC_AMI_MISSION  = "/ami/mission"
TOPIC_DV_STATUS    = "/dv/status"
TOPIC_CTRL_CMD     = "/ctrl/cmd"
TOPIC_FINISH_BRAKE = "/finish_brake"
SERVICE_FORCE_EBS  = "/force_ebs"

# Heartbeat liveness bound (seconds) — the canonical value shared by both
# directions. Each byte topic above is the other side's liveness heartbeat;
# a side reconciles/trips to its safe state when the other's heartbeat has
# been silent for longer than this. At the >= 10 Hz publish cadence that is
# 4 missed cycles (jitter-safe), and it sits strictly under
# HEARTBEAT_STALE_CAP_S — the FS-Rules T11.9.4 bound for detecting a lost
# safety-critical message and entering the safe state.
#
# This is the single source of truth. The uDV firmware MUST mirror it as
# DV_STATUS_STALE_MS in dv_interface.h (currently 400 ms) — the two repos
# have no build-time link, so keep them in lockstep by hand. The pipeline
# test suite pins HEARTBEAT_STALE_S <= HEARTBEAT_STALE_CAP_S; the firmware
# static_asserts DV_STATUS_STALE_MS < DV_STATUS_STALE_CAP_MS.
#
# Detection budget: each watcher only evaluates staleness on its own tick,
# so its worst-case detection latency is HEARTBEAT_STALE_S + one tick
# period — and THAT sum, not the bare window, must stay under the cap.
# Firmware side: AppTask loops ~1 ms, trivially fine. Pipeline side: the
# reconcile tick (_RECONCILE_HZ in mission_control_node) is sized so
# window + tick leaves real reaction margin, pinned by
# test_detection_budget_leaves_reaction_margin.
HEARTBEAT_STALE_CAP_S = 0.5    # FS-Rules T11.9.4 detect-and-safe cap
HEARTBEAT_STALE_S     = 0.4    # == firmware DV_STATUS_STALE_MS (400 ms)

# Sim operator panel — the AMI board + RES buttons stand-in. SIM-ONLY
# (Linux↔Linux): the backend/CLI drive sim_supervisor's emulated AS state
# machine through these; the real car has no equivalent (the AMI board /
# RES produce /ami/mission + the AS transitions directly). /sim/mission
# carries an AMI INDEX (not a registry id) so it forwards straight onto
# /ami/mission. The intent byte values are mirrored by
# as_state_machine.OperatorIntent.
TOPIC_SIM_MISSION = "/sim/mission"
TOPIC_SIM_INTENT  = "/sim/intent"
TOPIC_SIM_ESTOP   = "/sim/estop"
SIM_INTENT_OFF    = 0   # disarmed
SIM_INTENT_READY  = 1   # armed / prepare (RES go not pressed)
SIM_INTENT_GO     = 2   # RES go — run


# AMI mission INDEX (uDV ws2812.c mission_colors) → pipeline registry
# mission_id (mode_registry: trackdrive=1, autocross=2, accel=3,
# skidpad=4, scruti=5). 0 = no autonomy mission / tear down. mission_control
# maps the raw /ami/mission index through this so the uDV stays dumb (no
# registry numbering baked into firmware). Moved here from the deleted
# car_supervisor/policy.py; CONFIRM against AMI firmware (AMI 5 "EVS/EBS
# test" + AMI 6 "Inspection" both map to scruti for now).
DEFAULT_AMI_TO_MISSION_ID: dict[int, int] = {
    0: 0,   # Manual        → no autonomy mission
    1: 3,   # Acceleration  → accel
    2: 4,   # Skidpad       → skidpad
    3: 2,   # Autocross     → autocross
    4: 1,   # Track drive   → trackdrive
    5: 5,   # EVS/EBS test  → scruti   (CONFIRM)
    6: 5,   # Inspection    → scruti   (CONFIRM)
    7: 0,   # Shutdown      → no mission
    8: 0,   # Aux1          → no mission
    9: 0,   # Aux2          → no mission
}


def ami_index_to_mission_id(
    ami_index: int,
    mapping: dict[int, int] | None = None,
) -> int:
    """Translate an AMI mission index to a pipeline registry mission_id.

    Returns 0 (no autonomy mission / tear down) for any index not in the
    mapping, including the non-autonomy AMI slots (Manual, Shutdown,
    Aux). Never raises — an out-of-range index is treated as "no
    mission" so a glitchy /ami/mission can't start an unintended run.
    """
    table = DEFAULT_AMI_TO_MISSION_ID if mapping is None else mapping
    return int(table.get(int(ami_index), 0))


# Inverse map (registry mission_id → AMI index), first-wins so each
# runnable mission gets one canonical AMI index. Used ONLY by the sim
# operator panel (backend/CLI): users pick a registry mission, but
# /ami/mission must carry the AMI index in both sim and car so
# mission_control maps it identically. On the real car the AMI board
# produces the index directly — this inverse is never used there.
DEFAULT_MISSION_ID_TO_AMI_INDEX: dict[int, int] = {}
for _ami, _mid in DEFAULT_AMI_TO_MISSION_ID.items():
    if _mid != 0 and _mid not in DEFAULT_MISSION_ID_TO_AMI_INDEX:
        DEFAULT_MISSION_ID_TO_AMI_INDEX[_mid] = _ami


def mission_id_to_ami_index(
    mission_id: int,
    mapping: dict[int, int] | None = None,
) -> int:
    """Translate a registry mission_id to a sim AMI index for /sim/mission.

    Returns 0 (Manual / no autonomy mission) for mission_id 0 or any id
    without a runnable AMI slot. Round-trips through
    ami_index_to_mission_id for the runnable missions.
    """
    table = DEFAULT_MISSION_ID_TO_AMI_INDEX if mapping is None else mapping
    return int(table.get(int(mission_id), 0))
