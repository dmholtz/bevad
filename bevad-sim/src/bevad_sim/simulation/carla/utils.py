from __future__ import annotations

import carla
import numpy as np


def convert_carla_vector_noflip(vec: carla.Vector3D) -> np.ndarray:
    """
    Convert a carla.Vector3D (x,y,z) to np.ndarray(x,y,z) without flipping the y-coordinate.
    """
    return np.array([vec.x, vec.y, vec.z], dtype=np.float64)


def convert_carla_vector(vec: carla.Vector3D | carla.Location, homogeneous: bool = False) -> np.ndarray:
    """Convert a carla.Vector3D or carla.Location (x,y,z) given in LHS to np.ndarray (x,y,z) in RHS."""
    if homogeneous:
        return np.array([vec.x, -vec.y, vec.z, 1.0], dtype=np.float64)
    else:
        return np.array([vec.x, -vec.y, vec.z], dtype=np.float64)


def convert_carla_rotation(rot: carla.Rotation) -> np.ndarray:
    """
    Convert a carla.Rotation (roll, pitch, yaw) in deg to np.ndarray (roll, pitch, yaw) in rad
    and account for the direction.
    """
    degrees = np.array(
        [
            rot.roll,  # CARLA: counterclockwise positive about x-axis (no sign change)
            -rot.pitch,  # CARLA: counterclockwise positive about y-axis, but LHS to RHS conversion (sign change)
            -rot.yaw,  # CARLA: clockwise positive about z-axis (sign change)
        ],
        dtype=np.float64,
    )
    return degrees * (np.pi / 180.0)


def carla_weather_to_dict(cw: carla.WeatherParameters):
    """Convert a carla.WeatherParameters object into a dict."""
    dikt = {}
    for el in dir(cw):
        if el[0] != "_" and el[0].islower():
            dikt[el] = getattr(cw, el)
    return dikt
