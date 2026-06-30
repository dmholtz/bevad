"""
This module provides classes and functionalities to interface the CARLA simulator with bevad_sim.

Classes:
    CARLASimulator: Integrates the CARLA simulator with OpenAI Gym, providing functionalities to configure, initialize, and run simulations with various sensors and scenarios.
    CARLAActionSpace: A custom gymnasium action space for the CARLA simulator that defines the actions as carla.VehicleControl.
    SensorSynchronizer: Synchronizes sensor measurements in CARLA, ensuring all registered sensors provide their measurements within a specified timeout period.
    CarlaSaveTickingWorldWrapper: A wrapper for the CARLA world to override the `tick` method, ensuring the simulation progresses only after all sensor observations are processed.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import random
import time
from functools import partial
from typing import Any

import carla  # type: ignore
import gymnasium as gym
import numpy as np
import py_trees
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # type: ignore
from srunner.scenariomanager.timer import GameTime
from srunner.tools.route_parser import RouteParser  # type: ignore

from bevad_sim.data_interface.configurator import Configurator
from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.simulation.carla.extended_route_scenario import ExtendedRouteScenario
from bevad_sim.simulation.carla.leaderboard_tools import StatisticsManager
from bevad_sim.simulation.carla.observer.carla_observer import CarlaObserver
from bevad_sim.simulation.carla.sensor import Gnss, Imu
from bevad_sim.simulation.carla.sensors.camera import CameraBase, CameraRGB
from bevad_sim.simulation.carla.sensors.lidar import Lidar
from bevad_sim.simulation.engine.simulator_interface import CoreGymEnv


class SensorSynchronizer:
    """
    A class to synchronize sensor measurements in CARLA.

    This class ensures that all registered sensors provide their measurements within a specified timeout period.
    It logs warnings if a sensor provides multiple readings or if the timeout period is exceeded.
    """

    def __init__(self, timeout=2, verbose=0) -> None:
        """
        Initializes the synchronizer with the given timeout and verbosity level.

        Args:
            timeout (int, optional): The timeout value for the simulator.
            verbose (int, optional): The verbosity level for logging.
        """

        self.timeout = timeout

        self._registered_sensors: set | None = None
        self._sensor_measurement_queue: queue.Queue | None = None
        self._configure_logger(verbose=verbose)

        self.reset()

    def _configure_logger(self, verbose=0) -> None:
        """Configure the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    def reset(self):
        """
        This method initializes an empty set for registered sensors and an empty queue for sensor measurements.
        """
        self._registered_sensors = set()
        self._sensor_measurement_queue = queue.Queue()

    def register_sensor(self, sensor_id):
        """
        Registers a sensor with the given sensor_id.

        Args:
            sensor_id (int): The unique identifier of the sensor to be registered.

        Raises:
            AssertionError: If the sensor_id is already registered.
        """
        assert sensor_id not in self._registered_sensors, "Cannot double register sensor"
        self._registered_sensors.add(sensor_id)

    def on_sensor_read(self, sensor_id):
        """
        Callback function that is triggered when a sensor generates a reading.

        Args:
            sensor_id (int): The unique identifier of the sensor that generated the reading.
        """
        self._sensor_measurement_queue.put(sensor_id)

    def _wait_for_measurements_completed(self):
        processed_sensors = set()
        try:
            while processed_sensors != self._registered_sensors:
                sensor_id = self._sensor_measurement_queue.get(timeout=self.timeout)

                if sensor_id in processed_sensors:
                    self.logger.warning(f"Warning, SensorSynchronizer double read of sensor {sensor_id}. Skipping")
                    continue

                processed_sensors.add(sensor_id)
        except queue.Empty:
            unprocessed_sensors = self._registered_sensors - processed_sensors
            raise ValueError(
                f"Warning: SensorSynchronizer timed-out while waiting for {self.timeout} seconds to receive sensor measurements. Unprocessed sensors: {unprocessed_sensors}"
            )

        self._sensor_measurement_queue = queue.Queue()

    def empty(self) -> bool:
        """
        Check if the sensor measurement queue is empty.

        Returns:
            True if the sensor measurement queue is empty, False otherwise.
        """
        return self._sensor_measurement_queue.qsize() == 0


class CarlaSaveTickingWorldWrapper:
    """
    A wrapper class for the Carla world to override the `tick` method, ensuring the simulation progresses
    only after all sensor observations are processed to maintain strict synchronization.

    This wrapper is necessary because external modules (e.g., Scenario Runner) may directly invoke
    the `.tick()` method on the world object. By wrapping the world object, we can enforce
    safe, synchronized progression of the simulation, preventing any unsynchronized or unsafe activations.
    """

    def __init__(self, world: carla.world, synchronizer: SensorSynchronizer) -> None:
        self._world = world
        self._synchronizer = synchronizer

    def __getattr__(self, name):
        if hasattr(self._world, name):
            return getattr(self._world, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def tick(self) -> None:
        """
        Overrides tick with the save-tick function to make sure simulator is always synchronized
        """
        assert self._synchronizer.empty(), (
            "Sensor snychronizer must not have unprocessed readings before ticking the environment"
        )

        self._world.tick()
        self._synchronizer._wait_for_measurements_completed()

    def wait_for_tick(self) -> None:
        """
        This method bypasses blocking behavior when building new scenarios while the simulation runs,
        especially on long routes.

        The bypass is likely needed due to the fact that we tick and build new scenarios in the same
        thread, whereas leaderboard-2.0 runs the `build_scenarios_loop` concurrently.
        """
        pass


class CARLAActionSpace(gym.Space):
    """
    A custom gymnasium action space for CARLA simulator that defines the actions as carla.VehicleControl.
    """

    def __init__(self, seed: int | None = None):
        """
        Initializes the CARLAActionSpace.

        Args:
            seed: Seed for the random number generator.
        """
        super().__init__(shape=None, dtype=None, seed=seed)

    def sample(self) -> carla.VehicleControl:
        """
        Generates a random vehicle control action within the defined space.

        Returns:
            A randomly generated vehicle control action.
        """
        ctrl = carla.VehicleControl()
        ctrl.steer = np.random.uniform(-1.0, 1.0)
        ctrl.throttle = np.random.uniform(0.0, 1.0)
        ctrl.brake = np.random.uniform(0.0, 1.0)
        return ctrl

    def contains(self, x: dict[str, float]) -> bool:
        """
        Checks if a given vehicle control action is within the defined space.

        Args:
            x: The vehicle control action to check.

        Returns:
            True if the action is within the defined space, False otherwise.
        """
        throttle = x.get("throttle")
        steer = x.get("steer")
        brake = x.get("brake")

        if throttle is None or steer is None or brake is None:
            return False
        else:
            if (-1.0 <= steer <= 1.0) and (0.0 <= throttle <= 1.0) and (0.0 <= brake <= 1.0):
                return True
            else:
                return False


class CARLASimulator(CoreGymEnv, Configurator):
    """CARLASimulator class for simulating AD scenarios using the CARLA simulator.

    This class integrates the CARLA simulator with OpenAI Gym and provides functionalities to configure,
    initialize, and run simulations with various sensors and scenarios.

    Attributes:
        config (dict): Configuration parameters for the simulator.
        ticks_per_gymloop (int): Number of CARLA ticks per gym loop cycle.
        world (carla.World): The CARLA world object.
        observer (CarlaObserver): The observer for the CARLA environment.
        observation_space (ObservationSpace): The observation space for the environment.
        action_space (CARLAActionSpace): The action space for the environment.
        loaded_scenario (str, optional): The currently loaded scenario.
        sensors (list): List of sensors attached to the ego vehicle.
        sensor_synchronizer (SensorSynchronizer): Synchronizer for sensor data.
        reset_count (int): Counter for the number of environment resets.
        step_counter (int): Counter for the number of steps taken in the environment.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the CARLASimulator class.

        Args:
            config: Configuration parameters.
        """
        super().__init__()

        self.config = self.default_config()
        self.configure(config)
        self._configure_logger(verbose=self.config["verbose"])

        # verify configuration
        assert self.config["world_freq"] >= self.config["observer_freq"]
        assert self.config["world_freq"] % self.config["observer_freq"] == 0
        # it may be desirable to run CARLA at higher frequency than the gymloop for more realistic physics simulation
        self.ticks_per_gymloop = self.config["world_freq"] // self.config["observer_freq"]

        self.world = None
        self.observer = None
        self.input_action_space = CARLAActionSpace()

        self.loaded_scenario: str | None = None
        self.sensors: list[CameraBase] = []
        self.sensor_synchronizer = SensorSynchronizer(
            timeout=self.config["sensor_synchronizer_timeout"], verbose=self.config["verbose"]
        )

        self.reset_count: int = 0
        self.step_counter: int = 0

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """
        Returns the default configuration for the simulator.

        The configuration includes parameters for the agent, observer, environment, and simulation settings.

        Returns:
            dict[str, Any]: A dictionary containing the default configuration settings.

        Configuration Details:

        - **agent_config** (dict): Configuration for the agent.
            - **sensors** (dict): Dictionary of sensor configurations.
            - **ego_body** (dict): Configuration for the ego vehicle model.
                - **model** (str): Vehicle model (default: `"vehicle.audi.etron"`).

        - **observer_config** (dict): Default observer configuration from `CarlaObserver.default_config()`.

        - **town** (Optional[str]): The town map to be loaded. Defaults to `None`.

        - **default_background** (bool): Whether to use the default background environment. Defaults to `False`.

        - **custom_scenarios_path** (str): Path to custom scenario definitions.

        - **rct_window_size** (int): Window size for the reaction time computation. Defaults to `20`.

        - **world_freq** (int): Frequency of world updates (Hz). Defaults to `20`.

        - **policy_freq** (int): Frequency of policy updates (Hz). Defaults to `20`.

        - **observer_freq** (int): Frequency of observer updates (Hz). Defaults to `20`.

        - **verbose** (int): Verbosity level (higher values for more logs). Defaults to `0`.

        - **render** (bool): Whether to enable rendering. Defaults to `True`.

        - **carla_host** (str): Hostname of the CARLA simulator server. Defaults to `"localhost"`.

        - **carla_port** (int): Port number for CARLA communication. Defaults to `2000`.

        - **traffic_manager_port** (int): Port number for CARLA's traffic manager. Defaults to `8000`.

        - **timeout** (float): Timeout duration for CARLA connection (in seconds). Defaults to `200.0`.

        - **sensor_synchronizer_timeout** (int): Timeout for sensor synchronization. Defaults to `4`.
        """
        return {
            "agent_config": {
                "sensors": {},
                "ego_body": {"model": "vehicle.audi.etron"},
            },
            "observer_config": CarlaObserver.default_config(),
            "town": None,
            "default_background": False,
            "custom_scenarios_path": "bevad_sim.simulation.carla.custom_scenarios",
            "rct_window_size": 20,
            "world_freq": 20,
            "policy_freq": 20,
            "observer_freq": 20,
            "verbose": 0,
            "render": True,
            "carla_host": "localhost",
            "carla_port": 2000,
            "traffic_manager_port": 8000,
            "timeout": 200.0,
            "sensor_synchronizer_timeout": 4,
            "enforce_behavior": False,
        }

    def _init_world(self, town: str) -> tuple[carla.Client, carla.World, carla.TrafficManager]:
        """Initializes the CARLA simulation world with the specified town.

        This method sets up a new CARLA client, loads the specified town into the world,
        configures the world settings, and initializes the traffic manager. The world is
        wrapped with a custom ticking wrapper to ensure synchronization with sensors.

        Args:
            town: The name of the town to load into the CARLA world.

        Returns:
            The CARLA client object used to communicate with the server.
            The CARLA world object representing the simulation environment.
            The CARLA traffic manager object for controlling traffic behavior.

        Raises:
            carla.TimeoutException: If the client fails to connect to the CARLA server within the specified timeout.
            carla.WorldLoadException: If the specified town fails to load into the CARLA world.
        """
        client = carla.Client(self.config["carla_host"], self.config["carla_port"])
        client.set_timeout(self.config["timeout"])

        world = client.load_world(town, reset_settings=False)

        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.config["world_freq"],
            deterministic_ragdolls=True,
            spectator_as_ego=True,
            tile_stream_distance=650,
            actor_active_distance=650,
            no_rendering_mode=not self.config["render"],
        )
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(self.config["traffic_manager_port"])
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_random_device_seed(self.seed)

        world = CarlaSaveTickingWorldWrapper(world, synchronizer=self.sensor_synchronizer)
        world.tick()

        return client, world, traffic_manager

    def _create_sensors(self, ego: carla.Actor, sensor_config: list):
        """
        Create and configure sensors for the ego vehicle based on the provided sensor configuration.

        Args:
            ego: The ego vehicle for which the sensors are being created.
            sensor_config: A list of dictionaries, each containing the configuration for a sensor. Each dictionary must have a "type" key specifying the type of the sensor.

        Raises:
            NotImplementedError: If the sensor type is not yet implemented.

        """
        self.sensors = []

        acknowledge_fn = self.sensor_synchronizer.on_sensor_read
        for s in sensor_config:
            if s["type"] == "sensor.camera.rgb":
                sensor = CameraRGB(
                    self.world, ego, s, partial(self.observer.observe_tensor_fn, acknowledge_fn=acknowledge_fn)
                )
                self.sensor_synchronizer.register_sensor(f"{sensor.container_name}.{sensor.id}")
            elif s["type"] == "sensor.lidar.ray_cast":
                sensor = Lidar(
                    self.world, ego, s, partial(self.observer.observe_tensor_fn, acknowledge_fn=acknowledge_fn)
                )
                self.sensor_synchronizer.register_sensor(f"{sensor.container_name}.{sensor.id}")
            elif s["type"] == "imu":
                callback = partial(self.observer.observe_imu_fn, acknowledge_fn=acknowledge_fn)
                sensor = Imu(world=self.world, parent=ego, config=s, callback=callback)
                self.sensor_synchronizer.register_sensor(sensor.id)
            elif s["type"] == "gnss":
                callback = partial(self.observer.observe_gnss_fn, acknowledge_fn=acknowledge_fn)
                sensor = Gnss(world=self.world, parent=ego, config=s, callback=callback)
                self.sensor_synchronizer.register_sensor(sensor.id)
            elif s["type"] == "sensor.camera.depth":
                raise NotImplementedError()
            elif s["type"] == "sensor.camera.semantic":
                raise NotImplementedError()
            elif s["type"] == "sensor.collision":
                raise NotImplementedError()
            else:
                self.logger.error("Unknown sensor type: " + s["type"])
                continue
            self.logger.info(f"Spawned sensor: {s['id']}")
            self.world.tick()
            self.sensors.append(sensor)

    def _spawn_ego_vehicle(self, pose: carla.Transform) -> tuple[carla.Actor, int]:
        """Spawns the ego vehicle in the CARLA simulation environment.

        This method spawns the ego vehicle at the given pose using the vehicle model specified in the configuration.
        It also sets the role name of the vehicle to "hero" and creates the necessary sensors for the vehicle.

        Args:
            pose: The initial pose (position and orientation) where the ego vehicle will be spawned.

        Returns:
            The spawned ego vehicle actor.
            The actor ID of the spawned ego vehicle.
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(self.config["agent_config"]["ego_body"]["model"])
        vehicle_bp.set_attribute("role_name", "hero")

        response = self.client.apply_batch_sync([carla.command.SpawnActor(vehicle_bp, pose)])
        actor_id = response[0].actor_id
        ego = self.world.get_actor(actor_id)

        return ego, actor_id

    def _plan_route(self, grp: GlobalRoutePlanner) -> list:
        """Plans the route for the ego vehicle using the provided route planner.

        This method takes a GlobalRoutePlanner object and uses it to generate a
        detailed route for the ego vehicle based on predefined keypoints. The
        route is interpolated between each pair of consecutive keypoints to
        ensure a smooth path.

        Args:
            grp: An instance of the GlobalRoutePlanner used to compute the route between keypoints.

        Returns:
            A list of tuples where each tuple contains a waypoint transform and the connection type to the next waypoint.
        """
        route = []
        for i in range(len(self.route_config.keypoints) - 1):
            waypoint = self.route_config.keypoints[i]
            waypoint_next = self.route_config.keypoints[i + 1]
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            for wp, connection in interpolated_trace:
                route.append((wp.transform, connection))
        return route

    def _set_spectator(self, transform: carla.Transform) -> None:
        """Set the spectator's transform to a specified location and orientation.

        This method adjusts the spectator's viewpoint in the CARLA simulator by setting its
        transform to the given location and orientation. The spectator is positioned 30m
        above the given location and oriented to look directly downward.

        Args:
            transform (carla.Transform): The transform to set for the spectator. This includes
            the location and rotation in the CARLA world.
        """
        self.spectator.set_transform(
            carla.Transform(transform.location + carla.Location(z=30), carla.Rotation(pitch=-90))
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[CoreContainer, dict]:
        """This method resets the simulation environment to its initial state. It loads the necessary world,
        initializes the observer, spawns the ego vehicle, and sets up the route scenario. It also computes
        initial statistics and returns the initial observation.

        Args:
            seed: The seed for random number generation.
            options: A dictionary of options for resetting the environment. Must include the key "scenario_config" which specifies the path to the route file.

        Returns:
            A tuple containing the initial observation and an empty dictionary.

        Raises:
            ValueError: If the observer class returns None.

        Notes:
            - It ensures that the world is loaded only if it is not already loaded.
            - It cleans up the environment and resets the sensor synchronizer.
            - If a new route is specified, it parses the route file, plans the route, and initializes the statistics manager.
            - The ego vehicle is spawned at the initial pose, and the spectator view is set.
            - The observer starts observing, and the world is ticked to update the state.
            - Initial statistics are computed and stored in the observation.
        """
        start_time = time.time()
        self.logger.info("Resetting environment...")

        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = 0

        # check if a world is already loaded
        if not self.world:
            # load world

            town = self.config["town"]
            if not town:
                # parse scenario to determine town
                self.logger.info("Determine town")
                town = RouteParser.parse_routes_file(options["scenario_config"], "")[0].town

            self.logger.info(f"Loading town '{town}'")
            self.client, self.world, self.traffic_manager = self._init_world(town)
            self.spectator = self.world.get_spectator()
            self.town = town

        self._cleanup_environment()
        self.sensor_synchronizer.reset()

        if self.loaded_scenario != options["scenario_config"]:
            self.logger.info("\tLoading new route")
            self.route_config = RouteParser.parse_routes_file(options["scenario_config"], "")[0]
            assert self.route_config.town == self.town, (
                f"Town mismatch: {self.route_config.town} != {self.config['town']}"
            )
            grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1.0)
            self.route = self._plan_route(grp)
            self.ego_init_pose = self.route[0][0]
            self.ego_init_pose.location.z += 0.5

            self.statistics_manager = StatisticsManager("./simulation_results.json", "./live_results.txt")
            self.statistics_manager.create_route_data("Route", 0)

            self.loaded_scenario = options["scenario_config"]

        self.logger.info("\tSpawning ego vehicle")
        self.ego, self.ego_id = self._spawn_ego_vehicle(self.ego_init_pose)
        self._set_spectator(self.ego_init_pose)

        self._initialize_carla_data_provider()
        self._initialize_route_scenario()

        # build observer
        self.observer = CarlaObserver(
            self.town, self.world, self.route, options["scenario_config"], self.config["observer_config"]
        )
        self._create_sensors(self.ego, self.config["agent_config"]["sensors"])

        self.observer.start_observing()
        self.world.tick()
        obs = self.observer.finish_observing()

        # compute statistics
        self.statistics_manager.compute_route_statistics(0)
        self.route_record = self.statistics_manager._results.checkpoint.records[0]

        lb_metrics = {
            **self.route_record.infractions,
            **self.route_record.scores,
            "num_infractions": self.route_record.num_infractions,
        }

        obs.step_meta.info = [[{"lb_metrics": lb_metrics}]]

        self.step_counter = 0
        self.reset_count += 1

        end_time = time.time()
        self.logger.info(f"Reset No.: {self.reset_count} completed in {end_time - start_time:.4f} seconds")

        return obs, {}

    def _cleanup_environment(self) -> None:
        """
        Clean up the environment by removing scenarios, sensors, and actors.

        This method performs the following steps:
        - Removes scenarios:
           - Terminates the route scenario.
           - Destroys all parked actors associated with the route scenario.
           - Removes all actors from the route scenario.

        - Removes sensors:
           - Stops each sensor and checks if it is alive.
           - Collects commands to destroy each alive sensor actor.
           - Applies the batch of destroy commands synchronously.

        - Calls the cleanup method of CarlaDataProvider.
        """
        try:
            self.logger.info("\tRemoving Scenarios")
            self.route_scenario.terminate()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.route_scenario._parked_ids])
            self.route_scenario.remove_all_actors()
        except AttributeError:
            pass

        self.logger.info("\tRemoving Sensors")
        batch = []
        for s in self.sensors:
            actor = s.sensor
            actor.stop()
            if actor is not None and actor.is_alive:
                batch.append(carla.command.DestroyActor(actor))
        with contextlib.suppress(AttributeError):
            self.client.apply_batch_sync(batch)

        self.logger.info("\tCarlaDataProvider cleanup")
        CarlaDataProvider.cleanup()

    def _initialize_carla_data_provider(self) -> None:
        """
        This method sets up the CarlaDataProvider by executing the following steps:
            - Sets the world in CarlaDataProvider to the current simulation world.
            - Sets the client in CarlaDataProvider to the current simulation client.
            - Configures the traffic manager port in CarlaDataProvider.
            - Adds the ego vehicle to the CarlaDataProvider's actor pool.
            - Registers the ego vehicle actor with its initial pose in CarlaDataProvider.
        """
        CarlaDataProvider._rng = np.random.RandomState(self.seed)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(self.config["traffic_manager_port"])
        CarlaDataProvider._carla_actor_pool[self.ego_id] = self.ego  # type: ignore[has-type]
        CarlaDataProvider.register_actor(self.ego, self.ego_init_pose)  # type: ignore[has-type]

    def _initialize_route_scenario(self) -> None:
        """
        This method sets up the route scenario by performing the following steps:
            - Disables runtime initialization mode in CarlaDataProvider.
            - Creates an instance of ExtendedRouteScenario with the provided configuration.
            - Removes any existing scenario from the statistics manager.
            - Sets the newly created route scenario in the statistics manager.
        """
        self.logger.info("\tInitializing RouteScenario")
        CarlaDataProvider.set_runtime_init_mode(False)
        self.route_scenario = ExtendedRouteScenario(
            self.world,
            self.route_config,
            default_background=self.config["default_background"],
            custom_scenarios_path=self.config["custom_scenarios_path"],
            rct_window_size=self.config["rct_window_size"],
            debug_mode=0,
            criteria_enable=True,
            enforce_behavior=self.config["enforce_behavior"],
        )
        self.statistics_manager.remove_scenario()
        self.statistics_manager.set_scenario(self.route_scenario)

    def step(self, action: CoreContainer) -> tuple[CoreContainer, float, bool, bool, dict]:
        """This method simulates a single step in the CARLA environment using the provided action.
        It updates the environment state, computes statistics, and returns the observation, reward,
        and other information.

        Args:
            action: The action to take, which includes steering, throttle, and brake values.

        Returns:
            The observation after performing the action.
            The reward obtained from the action (currently always 0.0).
            A flag indicating if the episode has ended (currently always False).
            A flag indicating if the episode was truncated (currently always False).
            Additional information (currently an empty dictionary).

        Raises:
            AssertionError: If the environment is not initialized (i.e., `reset()` has not been called).

        Notes:
            - It performs multiple simulation ticks per gym loop cycle.
            - The observer is used to gather observations from the environment.
            - Route statistics are computed and stored in the observation.
            - The observation and action are stored for external access.
        """
        assert action.action is not None, "CoreContainer must contain a action"
        assert action.action.simulator_control is not None, "Action must contain a simulator_control"
        act = action.action.simulator_control[0][0]
        assert self.input_action_space.contains(act), (
            f"Action {act} is not in the input action space {self.input_action_space}"
        )
        act = carla.VehicleControl(
            throttle=act["throttle"],
            steer=act["steer"],
            brake=act["brake"],
        )

        assert hasattr(self, "ego"), "Environment not initialized. Call reset() first."
        self.logger.info(
            f"Step No.: {self.step_counter} - Action: [{act.steer:.2f}, {act.throttle:.2f}, {act.brake:.2f}] - Step per seconds: {1 / (time.time() - self.step_time) if self.step_counter > 0 else 0.0:.2f}"
        )
        self.step_time: float = time.time()

        for i in range(self.ticks_per_gymloop):
            # only observe the world state of the last tick in this gym loop cycle
            if i == self.ticks_per_gymloop - 1:
                self.observer.start_observing()

            self._simulate_step(act)
            self._set_spectator(self.ego.get_transform())  # type: ignore[has-type]

        obs = self.observer.finish_observing()
        self.step_counter += 1

        # compute statistics
        self.statistics_manager.compute_route_statistics(0)
        self.route_record = self.statistics_manager._results.checkpoint.records[0]  # type: ignore[has-type]

        lb_metrics = {
            **self.route_record.infractions,  # type: ignore[has-type]
            **self.route_record.scores,  # type: ignore[has-type]
            "num_infractions": self.route_record.num_infractions,  # type: ignore[has-type]
        }

        obs.step_meta.info = [[{"lb_metrics": lb_metrics}]]

        return obs, 0.0, False, False, {}

    def _simulate_step(self, ctrl: carla.VehicleControl) -> None:
        """This method performs a single simulation step in the CARLA environment by applying the given vehicle control
        command, updating the scenario, and ticking the world.

        Args:
            ctrl: The control command for the vehicle.

        Steps:
            1. Build and update the route scenarios for the ego vehicle.
            2. Spawn parked vehicles around the ego vehicle.
            3. Apply the vehicle control command to the ego vehicle.
            4. Retrieve the current simulation timestamp and update the game time.
            5. Update the CARLA data provider with the latest tick.
            6. Set the vehicle control command in the PyTrees blackboard.
            7. Tick the scenario tree if it has not yet succeeded.
            8. Advance the simulation by one tick.
        """
        self.route_scenario.build_scenarios(self.ego, debug=0)  # type: ignore[has-type]
        self.route_scenario.spawn_parked_vehicles(self.ego)  # type: ignore[has-type]

        self.client.apply_batch_sync([carla.command.ApplyVehicleControl(self.ego, ctrl)])  # type: ignore[has-type]
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()
        py_trees.blackboard.Blackboard().set("AV_control", ctrl)
        if self.route_scenario.scenario_tree.status != py_trees.common.Status.SUCCESS:
            self.route_scenario.scenario_tree.tick_once()
        self.world.tick()

    def render(self):
        """
        Render the current state of the simulator.

        This method is intended to be overridden by subclasses or wrappers
        to provide specific rendering functionality.
        By default, it does nothing.
        """
        pass

    def close(self) -> None:
        """
        Close the CARLASimulator.

        This method performs the following steps to properly close and clean up the CARLASimulator environment:

        - Destroys the spectator actor to free up resources.
        - Cleans up the environment by removing scenarios, sensors, and actors.
        """
        self.logger.info("Closing Environment")
        self.logger.info("\tDestroying Spectator")
        self.spectator.destroy()
        self._cleanup_environment()

        # Always disable sync mode before the script ends to prevent the server blocking whilst waiting for a tick
        self.logger.info("\tSetting Asynchronous Mode")
        world = self.client.get_world()
        settings = carla.WorldSettings(
            synchronous_mode=False,
        )
        world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(False)
        self.traffic_manager.shut_down()

        self.logger.info("\tClosing Logger")
        self.close_logger()
