from abc import ABC, abstractmethod
from typing import Any, Callable

import carla
import numpy as np

from bevad_sim.simulation.carla.utils import convert_carla_vector


class CarlaBaseSensor(ABC):
    """ "Base class for managing CARLA sensors."""

    def __init__(
        self,
        *,
        world: carla.World,
        parent: carla.Actor,
        config: dict,
        callback: Callable[[Any], None],
    ):
        # save the sensor id
        self.id = config["id"]

        # save the function for calling the observer
        self._observer_callback = callback

        # set up and spawn the sensor as specified by the config
        transform = self._extract_transform(config)
        bp_attributes = self._extract_attributes(config)
        self._setup_and_spawn(world=world, parent=parent, transform=transform, bp_attributes=bp_attributes)

    @property
    @abstractmethod
    def blueprint(self) -> str:
        """The name of the CARLA sensor blueprint."""
        raise NotADirectoryError

    @property
    def sensor(self) -> carla.Actor:
        """The CARLA sensor actor."""
        return self._sensor

    def _extract_attributes(self, config: dict) -> dict:
        """Default implementation for extracting blueprint attributes from the config."""
        return {}

    def _extract_transform(self, config: dict) -> carla.Transform:
        """Default implementation for extracting the sensor mounting transform from the config."""
        return carla.Transform(
            carla.Location(x=config["x"], y=config["y"], z=config["z"]),
            carla.Rotation(roll=config["roll"], pitch=config["pitch"], yaw=config["yaw"]),
        )

    def _setup_and_spawn(
        self, world: carla.World, parent: carla.Actor, transform: carla.Transform, bp_attributes: dict
    ):
        blueprint_library = world.get_blueprint_library()
        cam_bp = blueprint_library.find(self.blueprint)

        for name, value in bp_attributes.items():
            cam_bp.set_attribute(name, value)

        self._sensor = world.spawn_actor(cam_bp, transform, attach_to=parent)
        self._sensor.listen(self._sensor_callback)

    @abstractmethod
    def _sensor_callback(self, measurement: carla.SensorData):
        """The callback function which is passed to the CARLA sensor actor via the listen() method."""
        raise NotImplementedError()


class Gnss(CarlaBaseSensor):
    """Manages CARLA GNSS sensor."""

    @property
    def blueprint(self) -> str:
        return "sensor.other.gnss"

    def _extract_attributes(self, config: dict) -> dict:
        """Build blueprint attributes given the config dict."""
        return {}

    def _sensor_callback(self, measurement: carla.GnssMeasurement):
        self._observer_callback(
            {"id": self.id, "gps": np.array([[[measurement.latitude, measurement.longitude, measurement.altitude]]])}
        )


class Imu(CarlaBaseSensor):
    """Manages CARLA IMU sensor."""

    @property
    def blueprint(self) -> str:
        return "sensor.other.imu"

    def _extract_attributes(self, config: dict) -> dict:
        """Build blueprint attributes given the config dict."""
        return {}

    def _sensor_callback(self, measurement: carla.IMUMeasurement):
        """Receives a IMU measurement and converts it into bevad_sim conventions.

        Parameters
        ----------
        measurement: carla.IMUMeasurement
            - compass: float
                Given in rad. Represents a left-handed rotation w.r.t. the north.
            - angular_velocity: carla.Vector
                Given in rad/s. Despite being a vector, rotation directions follow usual CARLA
                conventions.
            - acceleration: carla.Vector
                Given in ms^-2 and in left-handed coordinates.

        Illustration of right-handed cordinate system and the compass.

        y (NORTH)
        ^
        |
        o--> x (EAST)
        """
        gyro = measurement.gyroscope
        self._observer_callback(
            {
                "id": self.id,
                # flip the left-handed rotation around z-axis
                "compass": np.array([[[-measurement.compass]]]),
                # see convert_carla_rotation for rotation axis definition
                "angular_velocity": np.array([[[gyro.x, -gyro.y, -gyro.z]]]),
                # usual LHS -> RHS conversion
                # TODO: check why CARLA reports positive z-accleration
                "acceleration": np.array([[convert_carla_vector(measurement.accelerometer)]]),
            }
        )
