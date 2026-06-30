from __future__ import annotations

import secrets
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import carla
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.data_interface.episode_map import MapContainer
from bevad_sim.data_interface.episode_meta import EpisodeMeta
from bevad_sim.data_interface.odometry import Odometry
from bevad_sim.data_interface.routing_information import RoutingInformation
from bevad_sim.data_interface.step_meta import StepMeta
from bevad_sim.data_interface.tce import TrafficControlElements
from bevad_sim.data_interface.tensor_observation import CameraObservation, LidarObservation, TensorObservation
from bevad_sim.data_interface.world_state import WorldState
from bevad_sim.simulation.carla.leaderboard_tools import downsample_route
from bevad_sim.simulation.carla.observer.extractor import build_world_state_extractor
from bevad_sim.simulation.carla.route_planner import RoutePlanner
from bevad_sim.simulation.carla.utils import convert_carla_vector, convert_carla_vector_noflip


class CarlaObserver:
    def __init__(self, town: str, world: carla.World, planned_route, scenario_file: Optional[Path] = None, config={}):
        self.world = world

        self.extractor = build_world_state_extractor(town=town, world=world)

        self.config = config

        self._observation = None

        w_settings = world.get_settings()

        # ignore incoming callbacks iff _paused
        self._paused = True

        self._lock = threading.Lock()

        # routing information
        full_routing = self._convert_route(planned_route, waypoint_dist=None)
        if (waypoint_dist := self.config["route_downsample_dist"]) is not None:
            self._observable_routing = self._convert_route(planned_route, waypoint_dist=waypoint_dist)
        else:
            self._observable_routing = full_routing
        self.navigation_goal_selector = RoutePlanner(self._observable_routing, min_distance=7.5, max_distance=50)

        # build episode-level metadata
        if scenario_file:
            scenarios, weathers = self._read_scenario_file(scenario_file)
            if len(weathers) > 0 and "name" in weathers[0]:
                weather = weathers[0]["name"]
            else:
                weather = None

            # build an episode-id based on the scenario name and a random suffix
            scenario_type = scenario_file.name[: -len(".xml")]
            episode_id = f"{scenario_type}-{secrets.token_hex(2)}"

            # full meta data
            meta_dict = {"scenario_file": scenario_file.name, "weather_parameters": weathers}
        else:
            meta_dict, scenarios, weather = {}, None, None
            episode_id = secrets.token_hex(16)

        # ego
        ego_actor = CarlaDataProvider.get_hero_actor()
        if ego_actor is None:
            raise RuntimeError("No hero actor found in the world, cannot build observer.")
        ego_extent = 2 * convert_carla_vector_noflip(ego_actor.bounding_box.extent)
        meta_dict["ego_vehicle"] = ego_actor.type_id

        self._episode_meta = EpisodeMeta(
            is_valid=np.ones((1,), dtype=bool),
            episode_id=[episode_id],
            scenario_type=[scenarios],
            region=[town],
            weather=[weather],
            nav_route=full_routing,
            frame_rate=np.array(
                [1.0 / w_settings.fixed_delta_seconds]
            ),  # warning: this only works if observer frequency matches tick rate
            ego_extent=ego_extent,
            meta=[meta_dict],
        )

    def observe_tensor_fn(self, tensor_obs: TensorObservation, acknowledge_fn: Callable[[str], None]):
        """Callback: Add a single tensor observation to the observed state and acknowledge."""

        with self._lock:
            assert len(tensor_obs.sensor_names[0]), (
                "A callback should deliver a single observation from exactly one sensor"
            )
            sensor_id = tensor_obs.sensor_names[0]

            # skip registration while observer is paused
            if not self._paused:
                assert self._observation is not None, "Observing has not started yet."
                self._tensor_observations[tensor_obs.container_name].append(tensor_obs)

            # acknowledge regardless whether observer is paused or not
            acknowledge_fn(f"{tensor_obs.container_name}.{sensor_id}")

    def observe_imu_fn(self, imu_measurement: dict[str, Any], acknowledge_fn: Callable[[str], None]):
        """Callback: Add information from IMU to the observed state and acknowledge."""

        with self._lock:
            # skip registration while observer is paused
            if not self._paused:
                # update the relevant odometry fields
                _odometry = (
                    self._observation.odometry if self._observation.odometry is not None else Odometry.create_empty()
                )
                _odometry.is_valid = np.ones((1, 1), dtype=bool)
                _odometry.compass = imu_measurement["compass"]
                _odometry.angular_velocity = imu_measurement["angular_velocity"]
                _odometry.acceleration = imu_measurement["acceleration"]
                self._observation.odometry = _odometry
            acknowledge_fn(imu_measurement["id"])

    def observe_gnss_fn(self, gnss_measurement: dict[str, Any], acknowledge_fn: Callable[[str], None]):
        """Callback: Add information from GNSS to the observed state and acknowledge."""

        with self._lock:
            # skip registration while observer is paused
            if not self._paused:
                # update the relevant odometry fields
                _odometry = (
                    self._observation.odometry if self._observation.odometry is not None else Odometry.create_empty()
                )
                _odometry.is_valid = np.ones((1, 1), dtype=bool)
                _odometry.gps = gnss_measurement["gps"]
                self._observation.odometry = _odometry

            acknowledge_fn(gnss_measurement["id"])

    def callback(self, measurement: Callable, acknowledge: Callable[[str], None]):
        """
        Callback to be called by all sensors, to add their measurements to the current observation

        :measurement: Measurement to add to observation.sensor_observation
        :acknowledge: Callback to route information back to calling instance, acknowledging that a sensor reading has been completed. Callback is called with sensor Id
        """
        with self._lock:
            self._observation.sensor_observation.add_sensor_measurement(measurement)
            print(
                "Callback: "
                + str(measurement.meta_data.sensor_id)
                + " Type: "
                + str(measurement.meta_data.sensor_type)
                + " timestamp: "
                + str(measurement.meta_data.timestamp)
            )

            acknowledge(measurement.meta_data.sensor_id)

    @classmethod
    def default_config(cls):
        config = {}

        config["visibility_pc_container"] = "point_cloud"  # the sensor used to compute visibility
        config["visibility_pc_sensor"] = "LIDAR_TOP"  # the sensor used to compute visibility
        config["visibility_fov_container"] = "rgb"  # the container name to check FOV visibility
        config["sensor_data_time_limit"] = 5.0
        config["max_num_frames"] = 50
        config["extraction_range"] = 200
        config["route_downsample_dist"] = 200  # default value from leaderboard 2.0

        return config

    def start_observing(self):
        with self._lock:
            assert not self._observation, "Observing has already started."
            self._paused = False

            self._observation = CoreContainer.create_empty()
            self._observation.is_valid = np.ones((1, 1), dtype=bool)
            self._observation.episode_meta = self._episode_meta
            self._tensor_observations = defaultdict(list)

    def finish_observing(self) -> CoreContainer:
        # TODO: check whether to wrapping with _lock is required
        pass

        assert self._observation is not None, "Observing has not started yet."

        ws, tces, ego_id, time_stamp, frame_id = self.extractor.extract_frame_snapshot()
        self._observation.world_state = ws
        self._observation.tce = tces
        self._observation.map_container = MapContainer(
            is_valid=np.ones((1,), dtype=bool), maps=[self.extractor.extract_map()]
        )
        self._observation.step_meta = StepMeta.create_empty()
        self._observation.step_meta.is_valid = np.ones((1, 1), dtype=bool)
        self._observation.step_meta.frame_ids = np.array([[frame_id]])
        self._observation.step_meta.timestamps = np.array([[time_stamp]])
        self._observation.world_state.ego_id = np.array([ego_id])

        # add batched TensorObservation
        for container, tensor_obs_list in self._tensor_observations.items():
            obs_type = type(tensor_obs_list[0])
            batched_obs = obs_type.build_sensor_batch(tensor_obs_list)
            setattr(self._observation, container, batched_obs)
        self._tensor_observations = defaultdict(list)

        # add routing information
        self._observation.routing_information = self.navigation_goal_selector.get_observation_routing_info(
            ws.transform[0, 0, 0, :3, 3]
        )

        # compute visibility of actors if LiDAR is available
        vis_pc_container = self.config["visibility_pc_container"]
        vis_pc_sensor = self.config["visibility_pc_sensor"]
        if (
            vis_pc_container in self._observation.lidar_observations
            and vis_pc_sensor in self._observation.lidar_observations[vis_pc_container].sensor_names
        ):
            lidar_obs = self._observation.lidar_observations[vis_pc_container]
            lidar_idx = lidar_obs.sensor_names[0].index(vis_pc_sensor)
            is_visible = self._compute_actor_visibility(
                self._observation.world_state, lidar_obs[:, :, lidar_idx : lidar_idx + 1]
            )
            self._observation.world_state.is_visible = is_visible

        # compute visibility of traffic control elements
        vis_fov_container = self.config["visibility_fov_container"]
        if (
            vis_fov_container in self._observation.camera_observations
            and vis_pc_container in self._observation.lidar_observations
            and vis_pc_sensor in self._observation.lidar_observations[vis_pc_container].sensor_names
        ):
            lidar_obs = self._observation.lidar_observations[vis_pc_container]
            lidar_idx = lidar_obs.sensor_names[0].index(vis_pc_sensor)
            cam_obs = self._observation.camera_observations[vis_fov_container]
            is_visible = self._compute_tce_visibility(
                self._observation.world_state,
                self._observation.tce,
                lidar_obs[:, :, lidar_idx : lidar_idx + 1],
                cam_obs,
            )
            self._observation.tce.is_visible = is_visible

        # reset current observation
        result_observation = self._observation
        self._observation = None
        self._paused = True

        return result_observation

    def _fix_observation(self):
        """Apply a series of fixes to the observation."""

        # fix nan in compass
        compass = self._observation.odometry.compass
        if compass is not None and np.any(np.isnan(compass)):
            world_tf_ego = self._observation.world_state.transform[0, 0, 0]
            ego_global = world_tf_ego @ np.array([1, 0, 0, 0])
            ego_yaw = float(np.arctan2(ego_global[1], ego_global[0]))
            compass = ego_yaw - np.pi / 2
            self._observation.odometry.compass = np.array([[[compass]]])

    def _convert_route(self, global_plan, waypoint_dist: float | int | None) -> RoutingInformation:
        """
        Convert a global plan from CARLA's GlobalRoutePlanner into a RoutingInformation instance.
        If waypoint_dist != None, the global plan is downsampled in advance.
        """

        if waypoint_dist is not None:
            ds_ids = downsample_route(global_plan, waypoint_dist)
            global_plan = [(global_plan[x][0], global_plan[x][1]) for x in ds_ids]

        waypoints = []
        commands = []
        for wp_tf, road_option in global_plan:
            pos = convert_carla_vector(wp_tf.location)
            waypoints.append(pos)
            commands.append(road_option.value)

        return RoutingInformation(
            is_valid=np.ones((1, 1), dtype=bool),
            target_lane_sequence=None,
            tactical_search_space=None,
            target_route=np.array(waypoints)[np.newaxis, ...],
            route_commands=[commands],
            navigation_goal=None,
        )

    def _read_scenario_file(self, scenario_file: Path):
        """Extract the list of scenario names and the weather from the given scenario XML file."""

        with open(scenario_file, "r") as f:
            scenario_xml_str = f.read()
        scenario_xml = ET.fromstring(scenario_xml_str)

        # gather all scenarios in a list
        scenarios = [scenario.get("type") for scenario in scenario_xml.findall(".//scenario")]

        # build a list of all weathers as JSON
        weather_elements = scenario_xml.findall(".//weather")
        weathers = [{attr: value for attr, value in we.attrib.items()} for we in weather_elements]

        return scenarios, weathers

    def _compute_actor_visibility(self, world_state: CoreContainer, lidar_obs: LidarObservation) -> None:
        # convert point cloud to ego-centric coordinates
        world_tf_ego = world_state.transform[0, 0, 0]
        ego_tf_lidar = np.linalg.inv(lidar_obs.extrinsics[0, 0, 0, :, :])
        world_tf_lidar = world_tf_ego @ ego_tf_lidar
        pc_lidar = lidar_obs.point_cloud[0, 0, 0, : lidar_obs.num_points[0, 0, 0], :4]
        pc_world = world_tf_lidar @ pc_lidar.T

        # transform point cloud into each actor's local coordinate system
        # and check whether points are inside the bounding box of the actor
        points_in_box = []
        for i in range(world_state.n_dim):
            box_tf_world = np.linalg.inv(world_state.transform[0, 0, i])
            half_size = world_state.extent[0, 0, i] / 2
            pc_box = (box_tf_world @ pc_world).T
            pc_box = pc_box[:, :3]

            in_box = np.all(np.logical_and(pc_box <= half_size, pc_box >= -half_size), axis=1)
            num_in_box = np.sum(in_box)
            points_in_box.append(num_in_box)

        # actor is visible if at least one point is inside the bounding box
        points_in_box = np.array(points_in_box)
        is_visible = points_in_box > 0

        # the ego actor is always invisible
        is_visible[0] = False

        return is_visible[np.newaxis, np.newaxis, :]

    def _compute_tce_visibility(
        self,
        world_state: WorldState,
        tce: TrafficControlElements,
        lidar_obs: LidarObservation,
        cam_obs: CameraObservation,
    ) -> None:
        # convert point cloud to ego-centric coordinates
        world_tf_ego = world_state.transform[0, 0, 0]
        ego_tf_world = np.linalg.inv(world_tf_ego)
        ego_tf_lidar = np.linalg.inv(lidar_obs.extrinsics[0, 0, 0, :, :])
        world_tf_lidar = world_tf_ego @ ego_tf_lidar
        pc_lidar = lidar_obs.point_cloud[0, 0, 0, : lidar_obs.num_points[0, 0, 0], :4]
        pc_world = world_tf_lidar @ pc_lidar.T

        # separate traffic signs and traffic lights
        is_sign = tce.category[0, 0] == tce.category_map["traffic_sign"]
        is_traffic_light = np.logical_not(is_sign)
        traffic_signs = tce[0, 0, is_sign]
        traffic_lights = tce[0, 0, is_traffic_light]

        # traffic signs: based on point cloud
        points_in_box = []
        for i in range(traffic_signs.n_dim):
            box_tf_world = np.linalg.inv(traffic_signs.transform[0, 0, i])
            half_size = traffic_signs.extent[0, 0, i] / 2
            pc_box = (box_tf_world @ pc_world).T
            pc_box = pc_box[:, :3]

            in_box = np.all(np.logical_and(pc_box <= half_size, pc_box >= -half_size), axis=1)
            num_in_box = np.sum(in_box)
            points_in_box.append(num_in_box)
        points_in_box = np.array(points_in_box)
        sign_is_visible = points_in_box > 0

        # traffic lights: based on camera FOV
        light_is_visible = []
        for i in range(traffic_lights.n_dim):
            world_tf_box = traffic_lights.transform[0, 0, i]
            visible = False
            for j in range(cam_obs.n_dim):
                # transform traffic light position to camera coordinates
                img_tf_cam = np.eye(4, dtype=np.float32)
                intrinsic = cam_obs.intrinsics[0, 0, j]
                img_tf_cam[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                cam_tf_ego = cam_obs.extrinsics[0, 0, j]
                img_tf_box = img_tf_cam @ cam_tf_ego @ ego_tf_world @ world_tf_box
                before_cam = img_tf_box[2, 3] > 0  # check if the traffic light is in front of the camera
                if before_cam:
                    divisor = img_tf_box[2, 3]
                    pixel = img_tf_box[:2, 3] / divisor  # normalize by z coordinate
                    h, w = cam_obs.height, cam_obs.width
                    visible = (0 <= pixel[0] < w) and (0 <= pixel[1] < h)

                if visible:
                    break

            light_is_visible.append(visible)
        light_is_visible = np.array(light_is_visible)

        # combine traffic signs and traffic lights visibility
        is_visible = np.zeros((1, 1, tce.n_dim), dtype=bool)
        is_visible[0, 0, is_sign] = sign_is_visible
        is_visible[0, 0, is_traffic_light] = light_is_visible

        return is_visible


def get_class():
    return CarlaObserver
