### TODO: Check if file is required, update or remove!
"""
This file defines custom CARLA scenarios for simulating background traffic on higway Town04 and braking tests.

The scenarios are implemented as subclasses of `BasicScenario`.

Each scenario initializes actors, defines behavior trees, and sets test criteria. The file is part of a larger
CARLA simulation framework and should be used in conjunction with the `ExtendedRouteScenario` module.
"""

import random

import carla  # type: ignore
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # type: ignore
from srunner.scenarios.basic_scenario import BasicScenario  # type: ignore


class RandomBackgroundActivity(BasicScenario):
    """A scenario that spawns random background vehicles in the CARLA world.

    Attributes:
        timeout (int): The maximum time allowed for the scenario to run.
    """

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        debug_mode=False,
        criteria_enable=True,
        terminate_on_failure=False,
        timeout=60,
    ):
        """Initializes the RandomBackgroundActivity scenario.

        Args:
            world (carla.World): The CARLA world instance.
            ego_vehicles (list): List of ego vehicles involved in the scenario.
            config (dict): Configuration dictionary for the scenario.
            debug_mode (bool, optional): Whether to enable debug mode. Defaults to False.
            criteria_enable (bool, optional): Whether to enable criteria evaluation. Defaults to True.
            terminate_on_failure (bool, optional): Whether to terminate the scenario on failure. Defaults to False.
            timeout (int, optional): The maximum time allowed for the scenario to run. Defaults to 60.
        """
        self.timeout = timeout
        super().__init__(
            "BGA",
            ego_vehicles,
            config,
            world,
            debug_mode,
            terminate_on_failure,
            criteria_enable,
        )

    def _initialize_actors(self, *_):
        """Initializes the actors for the scenario.

        This method spawns random vehicles at predefined spawn points and sets them to autopilot mode.
        """
        max_vehicles = 20
        spawn_points = [
            carla.Transform(
                carla.Location(x=random.uniform(-400, -220), y=random.uniform(27, 36), z=0.281942),
                carla.Rotation(pitch=0.000000, yaw=-0.368408, roll=0.000000),
            )
            for i in range(max_vehicles)
        ]

        # Draw the spawn point locations as numbers in the map
        # for i, spawn_point in enumerate(spawn_points):
        #    self.world.debug.draw_string(spawn_point.location, str(i), life_time=999)

        # Select some models from the blueprint library
        models = [
            "dodge",
            "audi",
            "model3",
            "mini",
            "mustang",
            "lincoln",
            "prius",
            "nissan",
            "crown",
            "impala",
        ]
        blueprints = []
        for vehicle in self.world.get_blueprint_library().filter("*vehicle*"):
            if any(model in vehicle.id for model in models):
                blueprints.append(vehicle)

        batch = []
        for i, spawn_point in enumerate(spawn_points):
            cmd = carla.command.SpawnActor(random.choice(blueprints), spawn_point)
            cmd.then(
                carla.command.SetAutopilot(
                    carla.command.FutureActor,
                    True,
                    CarlaDataProvider._traffic_manager_port,
                )
            )
            batch.append(cmd)
        responses = CarlaDataProvider._client.apply_batch_sync(batch, True)
        # self.world.tick()

        for resp in responses:
            if resp.has_error():
                pass
                # print("Error:", resp.error)
            else:
                actor = self.world.get_actor(resp.actor_id)
                self.other_actors.append(actor)
                CarlaDataProvider._carla_actor_pool[resp.actor_id] = actor
                CarlaDataProvider.register_actor(actor, actor.get_transform())
        # print("SPAWNED", len(self.other_actors))

    def _create_behavior(self):
        """Creates the behavior tree for the scenario.

        Returns:
            py_trees.composites.Parallel: A parallel behavior tree node.
        """
        return py_trees.composites.Parallel("BGA", py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

    def _create_test_criteria(self):
        """Creates the test criteria for the scenario.

        Returns:
            list: An empty list, as no specific criteria are defined for this scenario.
        """
        return []


class BrakeTestScenario(BasicScenario):
    """A scenario that tests the braking behavior of ego vehicles."""

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        debug_mode=False,
        criteria_enable=True,
        terminate_on_failure=False,
    ):
        """Initializes the BrakeTestScenario scenario.

        Args:
            world (carla.World): The CARLA world instance.
            ego_vehicles (list): List of ego vehicles involved in the scenario.
            config (dict): Configuration dictionary for the scenario.
            debug_mode (bool, optional): Whether to enable debug mode. Defaults to False.
            criteria_enable (bool, optional): Whether to enable criteria evaluation. Defaults to True.
            terminate_on_failure (bool, optional): Whether to terminate the scenario on failure. Defaults to False.
        """
        super().__init__(
            "BrakeTest",
            ego_vehicles,
            config,
            world,
            debug_mode,
            terminate_on_failure,
            criteria_enable,
        )

    def _initialize_actors(self):
        """Initializes the actors for the scenario.

        This method spawns vehicles at predefined spawn points and sets them to autopilot mode.
        """
        spawn_points = [
            carla.Transform(
                carla.Location(x=-180, y=26.5, z=10),
                carla.Rotation(pitch=0.000000, yaw=-0.368408, roll=0.000000),
            ),
            carla.Transform(
                carla.Location(x=-180, y=30, z=10),
                carla.Rotation(pitch=0.000000, yaw=-0.368408, roll=0.000000),
            ),
            carla.Transform(
                carla.Location(x=-180, y=33.5, z=10),
                carla.Rotation(pitch=0.000000, yaw=-0.368408, roll=0.000000),
            ),
            carla.Transform(
                carla.Location(x=-180, y=37, z=10),
                carla.Rotation(pitch=0.000000, yaw=-0.368408, roll=0.000000),
            ),
        ]

        # Draw the spawn point locations as numbers in the map
        for i, spawn_point in enumerate(spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(i), life_time=999)

        # Select some models from the blueprint library
        models = [
            "dodge",
            "audi",
            "model3",
            "mini",
            "mustang",
            "lincoln",
            "prius",
            "nissan",
            "crown",
            "impala",
        ]
        blueprints = []
        for vehicle in self.world.get_blueprint_library().filter("*vehicle*"):
            if any(model in vehicle.id for model in models):
                blueprints.append(vehicle)

        # tm_port = CarlaDataProvider._traffic_manager_port
        # tm = CarlaDataProvider.get_client().get_trafficmanager(tm_port)

        # Take a random sample of the spawn points and spawn some vehicles
        for i, spawn_point in enumerate(spawn_points):
            temp = self.world.spawn_actor(random.choice(blueprints), spawn_point)
            if temp is not None:
                # temp.set_autopilot(True, tm_port)
                # tm.distance_to_leading_vehicle(temp, 2)
                # tm.vehicle_percentage_speed_difference(temp, random.randint(-20, 30))
                # tm.random_right_lanechange_percentage(temp, 5)
                # tm.random_left_lanechange_percentage(temp, 5)

                self.other_actors.append(temp)
                CarlaDataProvider._carla_actor_pool[temp.id] = temp
                CarlaDataProvider.register_actor(temp, temp.get_transform())
        print("SPAWNED", len(self.other_actors))

    def _create_behavior(self):
        """Creates the behavior tree for the scenario.

        Returns:
            py_trees.composites.Parallel: A parallel behavior tree node.
        """
        return py_trees.composites.Parallel("BrakeTest", py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

    def _create_test_criteria(self):
        """Creates the test criteria for the scenario.

        Returns:
            list: An empty list, as no specific criteria are defined for this scenario.
        """
        return []
