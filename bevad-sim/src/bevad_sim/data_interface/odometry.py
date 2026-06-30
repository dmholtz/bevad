from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from bevad_sim.data_interface.base_entity import BaseEntity
from bevad_sim.data_interface.world_state import TransformsOperations as trop


@dataclass
class Odometry(BaseEntity):
    """
    Stores non-privileged odometry observations of the ego vehicle.
    All observations are measured by sensors and given in metric units with respect
    to the vehicle coordinate system according to ISO 8855.
    All attributes are optional.

    Attributes:

        cmd_acceleration: np.ndarray | None
            The acceleration command that was applied at that timestamp.
            Shape: (B>=1, T>=1, 1)
        cmd_steering_angle: np.ndarray | None
            The steering angle command  that was applied at that timestamp.
            Shape: (B>=1, T>=1, 1)
        tire_steering_angle: np.ndarray | None
            The steering angle of the front tires at that timestamp.
            Shape: (B>=1, T>=1, 1)
        gps: np.ndarray | None
            The GPS coordinates (latitude, longitude, altitude) of the ego vehicle.
            Shape: (B>=1, T>=1, 3)
        compass: np.ndarray | None
            The yaw-orientation of the ego vehicle with regard to the north in rad, measured by compass.
            Shape: (B>=1, T>=1, 1)
        speed: np.ndarray | None
            The longitudinal speed (scalar) in m/s of the ego vehcile, measured by a speedometer.
            Shape: (B>=1, T>=1, 1)
        angular_velocity: np.ndarray | None
            The angular velocity in rad/s of the vehicle, measured by a gyroscope.
            Shape: (B>=1, T>=1, 3)
        angular_acceleration: np.ndarray | None
            The angular acceleration in rad/s^2 of the vehicle, measured by a gyroscope.
            Shape: (B>=1, T>=1, 3)
        acceleration: np.ndarray | None
            The acceleration of the ego vehicle in ms^-2, measured by an accelerometer.
            TODO: add more specification about the coordinate system and z-axis orientation
            Shape: (B>=1, T>=1, 3)
        velocity: np.ndarray | None
            The acceleration of the ego vehicle in ms^-2, measured by an accelerometer.
            TODO: add more specification about the coordinate system and z-axis orientation
            Shape: (B>=1, T>=1, 3)
        transform: np.ndarray | None
            The transformation matrix of the ego vehicle in the world coordinate system.
            Shape: (B>=1, T>=1, 4, 4)
    """

    cmd_acceleration: (
        np.ndarray | torch.Tensor | None
    )  # field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    cmd_steering_angle: (
        np.ndarray | torch.Tensor | None
    )  # field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    tire_steering_angle: (
        np.ndarray | torch.Tensor | None
    )  # field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))

    gps: np.ndarray | torch.Tensor | None  # field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float64))

    compass: np.ndarray | torch.Tensor | None  # field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    speed: np.ndarray | torch.Tensor | None  # field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.float64))
    angular_velocity: (
        np.ndarray | torch.Tensor | None
    )  # field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float64))
    angular_acceleration: (
        np.ndarray | torch.Tensor | None
    )  # field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float64))
    acceleration: (
        np.ndarray | torch.Tensor | None
    )  # field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float64))
    velocity: np.ndarray | torch.Tensor | None  # field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float64))
    transform: (
        np.ndarray | torch.Tensor | None
    )  # field(default_factory=lambda: np.zeros((1, 1, 4, 4), dtype=np.float64))

    @property
    def dimensionality(self) -> int:
        return 2

    @property
    def t_dim(self) -> int:
        """Returns the size of the time dimension."""
        return self.is_valid.shape[1]

    @property
    def n_dim(self) -> None:
        """Returns the size of the element dimension. Returns none since object has no element dimension."""
        return None

    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        self._check_array_dim("cmd_acceleration", 2, None, ignore_list)
        self._check_array_dim("cmd_steering_angle", 2, None, ignore_list)
        self._check_array_dim("tire_steering_angle", 2, None, ignore_list)
        self._check_array_dim("gps", 2, (3,), ignore_list)
        self._check_array_dim("compass", 2, None, ignore_list)
        self._check_array_dim("speed", 2, None, ignore_list)
        self._check_array_dim("angular_velocity", 2, (3,), ignore_list)
        self._check_array_dim("angular_acceleration", 2, (3,), ignore_list)
        self._check_array_dim("acceleration", 2, (3,), ignore_list)
        self._check_array_dim("velocity", 2, (3,), ignore_list)
        self._check_array_dim("transform", 2, (4, 4), ignore_list)

    @classmethod
    def create_empty(cls):
        return Odometry(**{k: None for k in Odometry.__dataclass_fields__.keys()})

    @classmethod
    def create_zeros(cls, b=1, t=1) -> Odometry:
        """Creates an Odometry object with all fields initialized to zeros.
        Args:
            b (int, optional): Batch size. Defaults to 1.
            t (int, optional): Time steps. Defaults to 1.
        Returns:
            Odometry: An Odometry instance with all fields set to zero arrays of appropriate shapes.

        """
        is_valid = np.zeros((b, t, 1), dtype=np.uint8)
        gps = np.zeros((b, t, 3), dtype=np.float64)
        cmd_acceleration = np.zeros((b, t, 1), dtype=np.float64)
        cmd_steering_angle = np.zeros((b, t, 1), dtype=np.float64)
        tire_steering_angle = np.zeros((b, t, 1), dtype=np.float64)
        compass = np.zeros((b, t, 1), dtype=np.float64)
        speed = np.zeros((b, t, 1), dtype=np.float64)
        angular_velocity = np.zeros((b, t, 3), dtype=np.float64)
        angular_acceleration = np.zeros((b, t, 3), dtype=np.float64)
        acceleration = np.zeros((b, t, 3), dtype=np.float64)
        velocity = np.zeros((b, t, 3), dtype=np.float64)
        transform = np.zeros((b, t, 4, 4), dtype=np.float64)

        return Odometry(
            is_valid=is_valid,
            cmd_acceleration=cmd_acceleration,
            cmd_steering_angle=cmd_steering_angle,
            tire_steering_angle=tire_steering_angle,
            gps=gps,
            compass=compass,
            speed=speed,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            acceleration=acceleration,
            velocity=velocity,
            transform=transform,
        )

    def transform_to(self, trans_matrix: np.ndarray | torch.Tensor) -> Odometry:
        """
        Transforms the world state to a given reference frame.

        Args:
            reference_frame: The reference frame to transform to provided as
                4x4 homogenous transform. Expects a tensor of shape a tensor of
                shape (B=1, 4, 4).

        Returns:
            transformed_state: A new WorldState object transformed to the given
            reference frame.
        """

        assert self.shape[0] == 1, "For now transforming batched data is not supported! "
        assert self.transform is not None

        new_transform = trans_matrix @ self.transform

        return Odometry(
            is_valid=self.is_valid,
            cmd_acceleration=self.cmd_acceleration,
            cmd_steering_angle=self.cmd_steering_angle,
            tire_steering_angle=self.tire_steering_angle,
            gps=self.gps,
            compass=self.compass,
            speed=self.speed,
            angular_velocity=self.angular_velocity,
            angular_acceleration=self.angular_acceleration,
            acceleration=self.acceleration,
            velocity=self.velocity,
            transform=new_transform,
        )


class OdometryUtils:
    @classmethod
    def get_ego_state_from_odom(cls, odom: Odometry) -> np.ndarray:
        """Extracts an ego state (x,y,yaw, vel) from odometry.
        Args:
            odom (Odometry): The odometry data
        Returns:
            np.array: Anp.array of shape 4: (x,y,yaw, vel)

        """
        assert odom.transform is not None
        assert odom.speed is not None

        transform_np = odom.transform.numpy() if isinstance(odom.transform, torch.Tensor) else odom.transform
        ego_x, ego_y, ego_yaw = trop.get_xyyaw_from_transforms(transform_np[0, 0])
        ego_vel = odom.speed[0, 0]
        return np.array([ego_x, ego_y, ego_yaw, ego_vel]).flatten()
