# TODO refactor with sensor_base class


import carla
import numpy as np

from bevad_sim.data_interface.tensor_observation import LidarObservation
from bevad_sim.data_interface.world_state import TransformsOperations, inverse_transform_from_pos_rot
from bevad_sim.simulation.carla.utils import convert_carla_rotation, convert_carla_vector


class Lidar:
    def __init__(self, world, ego, config, callback):
        self.callback = callback

        self.id = config["id"]
        self.container_name = config["container_name"]

        # This is a first trivial implementation. If we see that many measurements overflow we should adapt
        time_per_step = world._world.get_settings().fixed_delta_seconds
        self.max_no_points = int(config["points_per_second"] * time_per_step)
        # We set the frequency depending on the simulator frequency to guarantee we always receive exactly one full lidar sweep
        rotation_frequency = 1 / time_per_step

        self.lidar_attributes = {
            "range": config["range"],
            "rotation_frequency": rotation_frequency,
            "channels": config["channels"],
            "upper_fov": config["upper_fov"],
            "lower_fov": config["lower_fov"],
            "horizontal_fov": config["horizontal_fov"],
            "points_per_second": config["points_per_second"],
            "atmosphere_attenuation_rate": config["atmosphere_attenuation_rate"],
            "dropoff_general_rate": config["dropoff_general_rate"],
            "dropoff_intensity_limit": config["dropoff_intensity_limit"],
            "dropoff_zero_intensity": config["dropoff_zero_intensity"],
        }

        self.pos_x = config["x"]
        self.pos_y = config["y"]
        self.pos_z = config["z"]
        self.roll = config["roll"]
        self.pitch = config["pitch"]
        self.yaw = config["yaw"]

        self.transform = carla.Transform(
            carla.Location(x=self.pos_x, y=self.pos_y, z=self.pos_z),
            carla.Rotation(roll=self.roll, pitch=self.pitch, yaw=self.yaw),
        )

        # Finally spawn actor
        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        for key, value in self.lidar_attributes.items():
            lidar_bp.set_attribute(key, str(value))

        self.sensor = world.spawn_actor(lidar_bp, self.transform, attach_to=ego)
        self.sensor.listen(self.sensor_callback)
        self.extrinsics = self.get_extrinsics()

    def get_extrinsics(self):
        """
        Returns the 4x4 homogeneous transformation lidar_T_car:
         - lidar is the lidar coordinate system according to ISO 8855
         - car is the vehicle coordinate system according to ISO 8855 with the origin being the bounding box center
        """

        # extract position / rotation of sensor w.r.t to its parent actor
        pos = convert_carla_vector(self.transform.location)
        rot = convert_carla_rotation(self.transform.rotation)
        lidar_tf_parent = inverse_transform_from_pos_rot(pos, rot)

        # extract position / rotation of the sensor's parent-actor's bounding box w.r.t. to the parent-actor transform
        sensor_parent_bb = self.sensor.parent.bounding_box
        sensor_parent_bb_loc = convert_carla_vector(sensor_parent_bb.location)
        sensor_parent_bb_rot = convert_carla_rotation(sensor_parent_bb.rotation)
        parent_tf_bb = TransformsOperations.get_transforms_pos_rot(sensor_parent_bb_loc, sensor_parent_bb_rot)

        # Combine
        lidar_tf_car = lidar_tf_parent @ parent_tf_bb
        return lidar_tf_car

    def get_meta_data(self, sensordata):
        raise ValueError("Who needs this")

    def get_sensor(self):
        return self.sensor

    def sensor_callback(self, lidar_measurement):
        point_cloud = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
        point_cloud = point_cloud.reshape(-1, 4)
        points_in_point_cloud = point_cloud.shape[0]

        # Points are currently (x,y,z, intensity). We make it homogeneous (x,y,z,1, intensity)
        coords = point_cloud[:, :3]
        # Rotate from carla to iso
        coords = coords * np.array([1, -1, 1], dtype=np.float32).reshape(1, 3)
        intensity = point_cloud[:, 3].reshape(-1, 1)
        point_cloud = np.hstack([coords, np.ones((points_in_point_cloud, 1), dtype=coords.dtype), intensity])

        points_to_padd = self.max_no_points - points_in_point_cloud
        if points_to_padd >= 0:
            point_cloud_padded = np.vstack(
                [point_cloud, np.zeros((points_to_padd, point_cloud.shape[1]), dtype=point_cloud.dtype)],
            )
        else:
            print(f"Warning: Too many points detected. Throwing away the last {-points_to_padd} points.")
            point_cloud_padded = point_cloud[: self.max_no_points]

        self.callback(
            LidarObservation(
                is_valid=np.ones((1, 1, 1), dtype=bool),
                container_name=self.container_name,
                _data=point_cloud_padded.reshape((1, 1, 1, *point_cloud_padded.shape)),
                timestamps=np.array([[[lidar_measurement.timestamp]]]),
                frame_ids=np.array([[[lidar_measurement.frame]]], dtype=int),
                sensor_names=[self.id],
                fileformat="npz",
                base_data_folder="",
                extrinsics=np.copy(self.extrinsics.reshape(1, 1, 1, *self.extrinsics.shape)),
                num_features=5,
                max_num_points=self.max_no_points,
                num_points=np.array([[[points_in_point_cloud]]], dtype=int),
            )
        )
