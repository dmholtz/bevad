from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.

    Args:
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray : A quaternion nd array in the format [w, x, y, z].
    """
    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]


def flatten_timestamps(timestamps: np.ndarray) -> List[float]:
    """
    Return a 1-D python list of float timestamps regardless of original shape.

    Args:
        timestamps (np.ndarray): An array of timestamps.

    Returns:
        List[float]: A flattened list of float timestamps.
    """
    if timestamps is None:
        return []
    return timestamps.squeeze().flatten().astype(float).tolist()


def to_scalar(value) -> float:
    """
    Convert a numpy scalar / array / python number to a plain python float.

    Args:
        value: A numpy scalar, array, or python number.

    Returns:
        float: The value converted to a plain python float.
    """
    return float(value.item()) if hasattr(value, "item") else float(value)
