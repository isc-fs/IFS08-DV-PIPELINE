"""Pure message builders for the deterministic-skidpad pose passthrough.

Kept out of cone_graph_slam_node (which needs gtsam/scipy to import) so the
passthrough contract is unit-testable off-node: given the EKF /odom message,
/slam/pose must carry the SAME pose, only relabelled to the map frame, and
map→odom must be identity. See skidpad-deterministic-design memory.
"""
from __future__ import annotations

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def passthrough_pose(odom_msg: Odometry, map_frame: str,
                     base_frame: str) -> Odometry:
    """Republish the EKF /odom pose as /slam/pose. map frame ≡ odom frame
    (identity correction), so the pose and twist are copied verbatim — the pose
    the operator trusts, unmodified. Twist is already body-frame in /odom, which
    is exactly what /slam/pose's child_frame_id (base_link) means."""
    out = Odometry()
    out.header.stamp = odom_msg.header.stamp
    out.header.frame_id = map_frame
    out.child_frame_id = base_frame
    out.pose.pose = odom_msg.pose.pose
    out.pose.covariance = odom_msg.pose.covariance
    out.twist.twist = odom_msg.twist.twist
    out.twist.covariance = odom_msg.twist.covariance
    return out


def identity_map_to_odom(stamp, map_frame: str,
                         odom_frame: str) -> TransformStamped:
    """Identity map→odom so the TF tree stays rooted (map→odom→base_link). The
    skidpad pipeline reads /odom and /slam/pose directly and needs no lookup;
    this is purely for TF consumers like rviz."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = map_frame
    t.child_frame_id = odom_frame
    t.transform.rotation.w = 1.0
    return t
