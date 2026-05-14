# Per-node lifecycle-transition progress event (#387).
#
# Published by mode_manager_node on `/mode_manager/progress` for every
# step of an activate_mode fan-out. mission_control_node subscribes
# and forwards a per-node string into StartMission feedback so the
# operator UI can show e.g. "configuring cone_detection_node" instead
# of a generic "warming_up" for the whole 15-25 s bring-up window.
#
# Decoupled topic (not StartMission feedback fields) so:
#   - Future subscribers (lichtblick dashboard, MC overlay) get
#     structured events without depending on the StartMission action
#     spec.
#   - Adding fields here doesn't break the StartMission wire format.

# Name of the node being driven. Matches the entries in
# mode_manager.AUTONOMY_LIFECYCLE_NODES.
string node_name

# lifecycle_msgs/msg/Transition IDs:
#   1 = CONFIGURE      2 = CLEANUP
#   3 = ACTIVATE       4 = DEACTIVATE
uint8 transition_id

# What just happened for this (node, transition) pair.
#   "starting" — about to call /<node>/change_state
#   "ok"       — change_state returned success
#   "skipped"  — pre-check found node already at the target state
#                (idempotency path: e.g. activate_mode called twice)
#   "failed"   — change_state returned success=False
#   "timeout"  — change_state didn't respond inside _CHANGE_STATE_TIMEOUT_S
string phase

# Diagnostic on failed/timeout. Empty otherwise.
string error

builtin_interfaces/Time stamp
