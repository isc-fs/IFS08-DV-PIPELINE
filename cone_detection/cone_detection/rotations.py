"""Rotation matrix utilities."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def vectors2matrix(v1, v2):
    """Rotation matrix that maps unit vector v1 onto unit vector v2.

    Both inputs must be normalised.
    """
    v_rot = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))
    v_norm = np.linalg.norm(v_rot)
    if v_norm == 0:
        # v1 ≡ ±v2 — return identity or flip.
        return np.eye(v1.shape[0]) if angle == 0 else -np.eye(v1.shape[0])
    v_rot /= v_norm
    return R.from_rotvec(v_rot * angle).as_matrix()
