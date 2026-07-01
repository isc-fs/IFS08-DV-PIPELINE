"""QoS profiles for the stock uDV ↔ mission_control interface.

Single source of truth for the QoS on the uDV heartbeat topics, so the
reconciler (`mission_control_node`) and the sim emulator
(`sim_supervisor_node`) — and any tooling — can never drift onto
incompatible profiles.

This lives in its own module rather than in `interface_contract.py`
because building a `QoSProfile` requires importing `rclpy`, and
`interface_contract` is deliberately kept dependency-free so it
unit-tests in plain pytest. Import the *names/bytes* from
`interface_contract`; import the *wire QoS* from here.

## Why BEST_EFFORT / VOLATILE

`/assi/state` and `/ami/mission` are steady ~10 Hz heartbeats: the uDV
firmware publishes them with the standard micro-ROS idiom — BEST_EFFORT
reliability, VOLATILE durability — matching all its sibling state topics.

DDS request-vs-offered matching means the reader QoS is not free to
differ:

  * A RELIABLE reader will NOT match a BEST_EFFORT writer.
  * A TRANSIENT_LOCAL reader will NOT match a VOLATILE writer.

so a "latched" (RELIABLE + TRANSIENT_LOCAL) subscription — which is what
these used to be — silently receives nothing from the real uDV, and the
reconciler sits at AS_OFF forever. BEST_EFFORT + VOLATILE on both the
reader and the emulator's writer is the only profile that matches the
firmware. Late-join is covered by the continuous heartbeat (both sides
publish every tick), not by durability, so VOLATILE loses nothing.

Keeping the emulator's publishers on this same profile is deliberate:
the sim then reproduces the car's exact QoS, so a regression to a
mismatched profile fails in sim instead of only on the vehicle.
"""
from __future__ import annotations

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


# Uplink heartbeat QoS for /assi/state + /ami/mission. Mirrors the uDV
# firmware's micro-ROS heartbeat idiom; see module docstring for the DDS
# matching rationale. depth 10 gives a small margin over the ~10 Hz
# cadence without ever latching.
UPLINK_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
