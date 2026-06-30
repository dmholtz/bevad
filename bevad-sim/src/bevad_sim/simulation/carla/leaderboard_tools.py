# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Code taken from the leaderboard repository: https://github.com/carla-simulator/leaderboard/tree/leaderboard-2.0
# RouteParser class copied from https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/route_parser.py
# RouteScenario class copied from https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/scenarios/route_scenario.py
# parked_vehicles list copied from https://github.com/carla-simulator/leaderboard/blob/a87a3419e9d2e0d36deb25f1f26c17edee2d1420/leaderboard/utils/parked_vehicles.py

"""
This module provides Challenge routes as standalone scenarios
"""

import importlib
import inspect
import json
import math
import os
import os.path
import pkgutil
import traceback
import xml.etree.ElementTree as ET  # noqa: N817
from typing import Dict, Type

import carla
import py_trees
import requests
from agents.navigation.local_planner import RoadOption
from dictor import dictor
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.lights_sim import RouteLightsBehavior
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest,
    CollisionTest,
    InRouteTest,
    MinimumSpeedRouteTest,
    OutsideRouteLanesTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitForBlackboardVariable
from srunner.scenariomanager.timer import RouteTimeoutBehavior
from srunner.scenariomanager.traffic_events import TrafficEventType
from srunner.scenariomanager.weather_sim import RouteWeatherBehavior
from srunner.scenarios.background_activity import BackgroundBehavior
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.tools.route_parser import ANGLE_THRESHOLD, DIST_THRESHOLD, convert_elem_to_transform

from bevad_sim.simulation.carla import parked_vehicles


class RouteParser:
    """
    Pure static class used to parse all the route and scenario configuration parameters.
    """

    @staticmethod
    def parse_routes_file(route_filename, routes_subset=""):
        """
        Returns a list of route configuration elements.
        :param route_filename: the path to a set of routes.
        :param single_route: If set, only this route shall be returned
        :return: List of dicts containing the waypoints, id and town of the routes
        """

        def get_routes_subset():
            """
            The route subset can be indicated by single routes separated by commas,
            or group of routes separated by dashes (or a combination of the two)"""
            subset_ids = []
            subset_groups = routes_subset.replace(" ", "").split(",")
            for group in subset_groups:
                if "-" in group:
                    # Group of route, iterate from start to end, making sure both ids exist
                    start, end = group.split("-")
                    found_start, found_end = (False, False)

                    for route in tree.iter("route"):
                        route_id = route.attrib["id"]
                        if not found_start and route_id == end:
                            raise ValueError(
                                f"Malformed route subset '{group}', found the end id before the starting one"
                            )
                        elif not found_start and route_id == start:
                            found_start = True
                        if not found_end and found_start:
                            if route_id in subset_ids:
                                raise ValueError(f"Found a repeated route with id '{route_id}'")
                            else:
                                subset_ids.append(route_id)
                            if route_id == end:
                                found_end = True

                    if not found_start:
                        raise ValueError(f"Couldn't find the route with id '{start}' inside the given routes file")
                    if not found_end:
                        raise ValueError(f"Couldn't find the route with id '{end}' inside the given routes file")

                else:
                    # Just one route, get its id while making sure it exists

                    found = False
                    for route in tree.iter("route"):
                        route_id = route.attrib["id"]
                        if route_id == group:
                            if route_id in subset_ids:
                                raise ValueError(f"Found a repeated route with id '{route_id}'")
                            else:
                                subset_ids.append(route_id)
                            found = True

                    if not found:
                        raise ValueError(f"Couldn't find the route with id '{group}' inside the given routes file")

            subset_ids.sort()
            return subset_ids

        route_configs = []
        tree = ET.parse(route_filename)
        if routes_subset:
            subset_list = get_routes_subset()
        for route in tree.iter("route"):
            route_id = route.attrib["id"]
            if routes_subset and route_id not in subset_list:
                continue

            route_config = RouteScenarioConfiguration()
            route_config.town = route.attrib["town"]
            route_config.name = f"RouteScenario_{route_id}"
            route_config.weather = RouteParser.parse_weather(route)

            # The list of carla.Location that serve as keypoints on this route
            positions = []
            for position in route.find("waypoints").iter("position"):
                positions.append(
                    carla.Location(
                        x=float(position.attrib["x"]), y=float(position.attrib["y"]), z=float(position.attrib["z"])
                    )
                )
            route_config.keypoints = positions

            # The list of ScenarioConfigurations that store the scenario's data
            scenario_configs = []
            for scenario in route.find("scenarios").iter("scenario"):
                scenario_config = ScenarioConfiguration()
                scenario_config.name = scenario.attrib.get("name")
                scenario_config.type = scenario.attrib.get("type")

                for elem in list(scenario):
                    if elem.tag == "trigger_point":
                        scenario_config.trigger_points.append(convert_elem_to_transform(elem))
                    elif elem.tag == "other_actor":
                        scenario_config.other_actors.append(ActorConfigurationData.parse_from_node(elem, "scenario"))
                    else:
                        scenario_config.other_parameters[elem.tag] = elem.attrib

                scenario_configs.append(scenario_config)
            route_config.scenario_configs = scenario_configs

            route_configs.append(route_config)

        return route_configs

    @staticmethod
    def parse_weather(route):
        """
        Parses all the weather information as a list of [position, carla.WeatherParameters],
        where the position represents a % of the route.
        """
        weathers = []

        weathers_elem = route.find("weathers")
        if weathers_elem is None:
            return [[0, carla.WeatherParameters(sun_altitude_angle=70, cloudiness=50)]]

        for weather_elem in weathers_elem.iter("weather"):
            route_percentage = float(weather_elem.attrib["route_percentage"])

            weather = carla.WeatherParameters(sun_altitude_angle=70, cloudiness=50)  # Base weather
            for weather_attrib in weather_elem.attrib:
                if hasattr(weather, weather_attrib):
                    setattr(weather, weather_attrib, float(weather_elem.attrib[weather_attrib]))
                elif weather_attrib != "route_percentage":
                    print(f"WARNING: Ignoring '{weather_attrib}', as it isn't a weather parameter")

            weathers.append([route_percentage, weather])

        weathers.sort(key=lambda x: x[0])
        return weathers

    @staticmethod
    def is_scenario_at_route(trigger_transform, route):
        """
        Check if the scenario is affecting the route.
        This is true if the trigger position is very close to any route point
        """

        def is_trigger_close(trigger_transform, route_transform):
            """Check if the two transforms are similar"""
            dx = trigger_transform.location.x - route_transform.location.x
            dy = trigger_transform.location.y - route_transform.location.y
            dz = trigger_transform.location.z - route_transform.location.z
            dpos = math.sqrt(dx * dx + dy * dy)

            dyaw = (float(trigger_transform.rotation.yaw) - route_transform.rotation.yaw) % 360

            return (
                dz < DIST_THRESHOLD
                and dpos < DIST_THRESHOLD
                and (dyaw < ANGLE_THRESHOLD or dyaw > (360 - ANGLE_THRESHOLD))
            )

        for route_transform, _ in route:  # noqa: SIM110
            if is_trigger_close(trigger_transform, route_transform):
                return True

        return False


class RouteScenario(BasicScenario):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"
    INIT_THRESHOLD = 500  # Runtime initialization trigger distance to ego (m)
    PARKED_VEHICLES_INIT_THRESHOLD = (
        INIT_THRESHOLD - 50
    )  # Runtime initialization trigger distance to parked vehicles (m)

    def __init__(self, world, config, debug_mode=0, criteria_enable=True, enforce_behavior: bool = False):
        """
        Setup all relevant parameters and create scenarios along route

        Args:
            enforce_behavior (bool): This flag enforces the behavior tree to be built, even if some scenarios fail to initialize.
                If True and a scenario fails to initialize, the simulation raises and exception.
                If False and a scenario fails to initialize, the scenario is skipped and the simulation continues.

        """
        self.client = CarlaDataProvider.get_client()
        self.config = config
        self.route = self._get_route(config)
        self.world = world
        self.map = CarlaDataProvider.get_map()
        self.timeout = 10000
        self.enforce_behavior = enforce_behavior

        self.all_scenario_classes = self._find_scenario_classes()

        self.ego_data = None

        self.scenario_triggerer = None
        self.behavior_node = None  # behavior node created by _create_behavior()
        self.criteria_node = None  # criteria node created by _create_test_criteria()

        self.list_scenarios = []
        self.occupied_parking_locations = []
        self.available_parking_locations = []

        scenario_configurations = self._filter_scenarios(config.scenario_configs)
        self.scenario_configurations = scenario_configurations
        self.missing_scenario_configurations = scenario_configurations.copy()

        ego_vehicle = CarlaDataProvider.get_hero_actor()
        if ego_vehicle is None:
            raise ValueError("Shutting down, couldn't spawn the ego vehicle")

        if debug_mode > 0:
            self._draw_waypoints(self.route, vertical_shift=0.1, size=0.1, downsample=10)

        self._parked_ids = []
        self._get_parking_slots()
        super(RouteScenario, self).__init__(  # noqa: UP008
            config.name,
            [ego_vehicle],
            config,
            world,
            debug_mode > 3,
            False,
            criteria_enable,
        )
        # Do it after the 'super', as we need the behavior and criteria tree to be initialized
        self.build_scenarios(ego_vehicle, debug=debug_mode > 0)

        # Set runtime init mode. Do this after the first set of scenarios has been initialized!
        CarlaDataProvider.set_runtime_init_mode(True)

    def _get_route(self, config):
        """
        Gets the route from the configuration, interpolating it to the desired density,
        saving it to the CarlaDataProvider and sending it to the agent

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        - debug_mode: boolean to decide whether or not the route poitns are printed
        """

        # Prepare route's trajectory (interpolate and add the GPS route)
        self.gps_route, self.route = interpolate_trajectory(config.keypoints)
        return self.route

    def _filter_scenarios(self, scenario_configs):
        """
        Given a list of scenarios, filters out does that don't make sense to be triggered,
        as they are either too far from the route or don't fit with the route shape

        Parameters:
        - scenario_configs: list of ScenarioConfiguration
        """
        new_scenarios_config = []
        for scenario_number, scenario_config in enumerate(scenario_configs):
            trigger_point = scenario_config.trigger_points[0]
            if not RouteParser.is_scenario_at_route(trigger_point, self.route):
                print(f"WARNING: Ignoring scenario '{scenario_config.name}' as it is too far from the route")
                continue

            scenario_config.route_var_name = f"ScenarioRouteNumber{scenario_number}"
            new_scenarios_config.append(scenario_config)

        return new_scenarios_config

    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at the first waypoint of the route"""
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor(
            "vehicle.lincoln.mkz_2020", elevate_transform, rolename="hero"
        )
        if not ego_vehicle:
            return

        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(elevate_transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))
        )

        self.world.tick()

        return ego_vehicle

    def _get_parking_slots(self, max_distance=100, route_step=10):
        """Spawn parked vehicles."""

        def is_close(slot_location):
            for i in range(0, len(self.route), route_step):
                route_transform = self.route[i][0]
                if route_transform.location.distance(slot_location) < max_distance:
                    return True
            return False

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        for route_transform, _ in self.route:
            min_x = min(min_x, route_transform.location.x - max_distance)
            min_y = min(min_y, route_transform.location.y - max_distance)
            max_x = max(max_x, route_transform.location.x + max_distance)
            max_y = max(max_y, route_transform.location.y + max_distance)

        # Occupied parking locations
        occupied_parking_locations = []
        for scenario in self.list_scenarios:
            occupied_parking_locations.extend(scenario.get_parking_slots())

        available_parking_locations = []
        map_name = self.map.name.split("/")[-1]
        available_parking_locations = getattr(parked_vehicles, map_name, [])

        # Exclude parking slots that are too far from the route
        for slot in available_parking_locations:
            slot_transform = carla.Transform(
                location=carla.Location(slot["location"][0], slot["location"][1], slot["location"][2]),
                rotation=carla.Rotation(slot["rotation"][0], slot["rotation"][1], slot["rotation"][2]),
            )

            in_area = (min_x < slot_transform.location.x < max_x) and (min_y < slot_transform.location.y < max_y)
            close_to_route = is_close(slot_transform.location)
            if not in_area or not close_to_route:
                available_parking_locations.remove(slot)
                continue

        self.available_parking_locations = available_parking_locations

    def spawn_parked_vehicles(self, ego_vehicle, max_scenario_distance=10):
        """Spawn parked vehicles."""

        def is_close(slot_location, ego_location):
            return slot_location.distance(ego_location) < self.PARKED_VEHICLES_INIT_THRESHOLD

        def is_free(slot_location):
            for occupied_slot in self.occupied_parking_locations:
                if slot_location.distance(occupied_slot) < max_scenario_distance:
                    return False
            return True

        new_parked_vehicles = []

        ego_location = CarlaDataProvider.get_location(ego_vehicle)
        if ego_location is None:
            return

        for slot in self.available_parking_locations:
            slot_transform = carla.Transform(
                location=carla.Location(slot["location"][0], slot["location"][1], slot["location"][2]),
                rotation=carla.Rotation(slot["rotation"][0], slot["rotation"][1], slot["rotation"][2]),
            )

            # Add all vehicles that are close to the ego and in a free space
            if is_close(slot_transform.location, ego_location) and is_free(slot_transform.location):
                mesh_bp = CarlaDataProvider.get_world().get_blueprint_library().filter("static.prop.mesh")[0]
                mesh_bp.set_attribute("mesh_path", slot["mesh"])
                mesh_bp.set_attribute("scale", "0.9")
                new_parked_vehicles.append(carla.command.SpawnActor(mesh_bp, slot_transform))
                self.available_parking_locations.remove(slot)

        # Add the actors to _parked_ids
        for response in CarlaDataProvider.get_client().apply_batch_sync(new_parked_vehicles):
            if not response.error:
                self._parked_ids.append(response.actor_id)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, waypoints, vertical_shift, size, downsample=1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for i, w in enumerate(waypoints):
            if i % downsample != 0:
                continue

            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green

            self.world.debug.draw_point(wp, size=size, color=color, life_time=self.timeout)

        self.world.debug.draw_point(
            waypoints[0][0].location + carla.Location(z=vertical_shift),
            size=2 * size,
            color=carla.Color(0, 0, 128),
            life_time=self.timeout,
        )
        self.world.debug.draw_point(
            waypoints[-1][0].location + carla.Location(z=vertical_shift),
            size=2 * size,
            color=carla.Color(128, 128, 128),
            life_time=self.timeout,
        )

    def build_scenarios(self, ego_vehicle, debug=False):
        """
        Initializes the class of all the scenarios that will be present in the route.
        If a class fails to be initialized, a warning is printed but the route execution isn't stopped
        """
        new_scenarios = []

        if self.ego_data is None:
            self.ego_data = ActorConfigurationData(ego_vehicle.type_id, ego_vehicle.get_transform(), "hero")

        # Part 1. Check all scenarios that haven't been initialized, starting them if close enough to the ego vehicle
        for scenario_config in self.missing_scenario_configurations:
            scenario_config.ego_vehicles = [self.ego_data]
            scenario_config.route = self.route

            try:
                scenario_class = self.all_scenario_classes[scenario_config.type]
                trigger_location = scenario_config.trigger_points[0].location

                ego_location = CarlaDataProvider.get_location(ego_vehicle)
                if ego_location is None:
                    continue

                # Only init scenarios that are close to ego
                if trigger_location.distance(ego_location) < self.INIT_THRESHOLD:
                    scenario_instance = scenario_class(self.world, [ego_vehicle], scenario_config, timeout=self.timeout)

                    # Add new scenarios to list
                    self.list_scenarios.append(scenario_instance)
                    new_scenarios.append(scenario_instance)
                    self.missing_scenario_configurations.remove(scenario_config)

                    self.occupied_parking_locations.extend(scenario_instance.get_parking_slots())

                    if debug:
                        scenario_loc = scenario_config.trigger_points[0].location
                        debug_loc = self.map.get_waypoint(scenario_loc).transform.location + carla.Location(z=0.2)
                        self.world.debug.draw_point(
                            debug_loc,
                            size=0.2,
                            color=carla.Color(128, 0, 0),
                            life_time=self.timeout,
                        )
                        self.world.debug.draw_string(
                            debug_loc,
                            str(scenario_config.name),
                            draw_shadow=False,
                            color=carla.Color(0, 0, 128),
                            life_time=self.timeout,
                            persistent_lines=True,
                        )

            except Exception as e:  # noqa: BLE001
                if self.enforce_behavior:
                    raise RuntimeError(f"Failed to initialize scenario '{scenario_config.name}': {e}") from e
                else:
                    print(f"\033[93mSkipping scenario '{scenario_config.name}' due to setup error: {e}")
                    if debug:
                        print(f"\n{traceback.format_exc()}")
                    print("\033[0m", end="")
                    self.missing_scenario_configurations.remove(scenario_config)
                    continue

        # Part 2. Add their behavior onto the route's behavior tree
        for scenario in new_scenarios:
            # Add behavior
            if scenario.behavior_tree is not None:
                self.behavior_node.add_child(scenario.behavior_tree)
                self.scenario_triggerer.add_blackboard(
                    [
                        scenario.config.route_var_name,
                        scenario.config.trigger_points[0].location,
                        scenario.name,
                    ]
                )

            # Add the criteria criteria
            scenario_criteria = scenario.get_criteria()
            if len(scenario_criteria) == 0:
                continue

            self.criteria_node.add_child(self._create_criterion_tree(scenario, scenario_criteria))

    # pylint: enable=no-self-use
    def _initialize_actors(self, config):  # noqa: ARG002
        """
        Set other_actors to the superset of all scenario actors
        """
        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Creates a parallel behavior that runs all of the scenarios part of the route.
        These subbehaviors have had a trigger condition added so that they wait until
        the agent is close to their trigger point before activating.

        It also adds the BackgroundActivity scenario, which will be active throughout the whole route.
        This behavior never ends and the end condition is given by the RouteCompletionTest criterion.
        """
        scenario_trigger_distance = DIST_THRESHOLD  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(
            name="Route Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )

        self.behavior_node = behavior
        scenario_behaviors = []
        blackboard_list = []

        # Add the behavior that manages the scenario trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance
        )
        behavior.add_child(scenario_triggerer)  # Tick the ScenarioTriggerer before the scenarios

        # register var
        self.scenario_triggerer = scenario_triggerer

        # Add the Background Activity
        behavior.add_child(BackgroundBehavior(self.ego_vehicles[0], self.route, name="BackgroundActivity"))

        behavior.add_children(scenario_behaviors)
        return behavior

    def _create_test_criteria(self):
        """
        Create the criteria tree. It starts with some route criteria (which are always active),
        and adds the scenario specific ones, which will only be active during their scenario
        """
        criteria = py_trees.composites.Parallel(name="Criteria", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        self.criteria_node = criteria

        # End condition
        criteria.add_child(RouteCompletionTest(self.ego_vehicles[0], route=self.route))

        # 'Normal' criteria
        criteria.add_child(OutsideRouteLanesTest(self.ego_vehicles[0], route=self.route))
        criteria.add_child(CollisionTest(self.ego_vehicles[0], name="CollisionTest"))
        criteria.add_child(RunningRedLightTest(self.ego_vehicles[0]))
        criteria.add_child(RunningStopTest(self.ego_vehicles[0]))
        criteria.add_child(MinimumSpeedRouteTest(self.ego_vehicles[0], self.route, checkpoints=4, name="MinSpeedTest"))

        # These stop the route early to save computational time
        criteria.add_child(
            InRouteTest(
                self.ego_vehicles[0],
                route=self.route,
                offroad_max=30,
                terminate_on_failure=True,
            )
        )
        criteria.add_child(
            ActorBlockedTest(
                self.ego_vehicles[0],
                min_speed=0.1,
                max_time=180.0,
                terminate_on_failure=True,
                name="AgentBlockedTest",
            )
        )

        return criteria

    def _create_weather_behavior(self):
        """
        Create the weather behavior
        """
        if len(self.config.weather) == 1:
            return  # Just set the weather at the beginning and done
        return RouteWeatherBehavior(self.ego_vehicles[0], self.route, self.config.weather)

    def _create_lights_behavior(self):
        """
        Create the street lights behavior
        """
        return RouteLightsBehavior(self.ego_vehicles[0], 100)

    def _create_timeout_behavior(self):
        """
        Create the timeout behavior
        """
        return RouteTimeoutBehavior(self.ego_vehicles[0], self.route)

    def _initialize_environment(self, world):
        """
        Set the weather
        """
        # Set the appropriate weather conditions
        world.set_weather(self.config.weather[0][1])

    def _create_criterion_tree(self, scenario, criteria):
        """
        We can make use of the blackboard variables used by the behaviors themselves,
        as we already have an atomic that handles their (de)activation.
        The criteria will wait until that variable is active (the scenario has started),
        and will automatically stop when it deactivates (as the scenario has finished)
        """
        scenario_name = scenario.name
        var_name = scenario.config.route_var_name
        check_name = f"WaitForBlackboardVariable: {var_name}"

        criteria_tree = py_trees.composites.Sequence(name=scenario_name)
        criteria_tree.add_child(WaitForBlackboardVariable(var_name, True, False, name=check_name))

        scenario_criteria = py_trees.composites.Parallel(
            name=scenario_name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        for criterion in criteria:
            scenario_criteria.add_child(criterion)
        scenario_criteria.add_child(WaitForBlackboardVariable(var_name, False, None, name=check_name))

        criteria_tree.add_child(scenario_criteria)
        criteria_tree.add_child(Idle())  # Avoid the indiviual criteria stopping the simulation
        return criteria_tree

    def _find_scenario_classes(self, scenarios_pkg_name="srunner.scenarios") -> Dict[str, Type]:
        "Find all scenario classes provided by srunner."

        scenarios_pkg = importlib.import_module(scenarios_pkg_name)
        scenario_classes = {}

        for _, module_name, is_pkg in pkgutil.iter_modules(scenarios_pkg.__path__):
            if is_pkg:
                continue

            # TODO: these modules are currently not importable
            if module_name in ("osc2_scenario",):
                continue

            scenario_module_name = f"{scenarios_pkg_name}.{module_name}"
            scenario_module = importlib.import_module(scenario_module_name)

            for mem_name, member in inspect.getmembers(scenario_module, inspect.isclass):
                assert isinstance(member, Type)

                if not issubclass(member, BasicScenario):
                    continue

                scenario_classes[mem_name] = member

        return scenario_classes

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self._parked_ids])
        self.remove_all_actors()


# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""


def autodetect_proxy():
    proxies = {}

    proxy_https = os.getenv("HTTPS_PROXY", os.getenv("https_proxy", None))
    proxy_http = os.getenv("HTTP_PROXY", os.getenv("http_proxy", None))

    if proxy_https:
        proxies["https"] = proxy_https
    if proxy_http:
        proxies["http"] = proxy_http

    return proxies


def fetch_dict(endpoint):
    data = None
    if endpoint.startswith(("http:", "https:", "ftp:")):
        proxies = autodetect_proxy()

        if proxies:
            response = requests.get(url=endpoint, proxies=proxies)
        else:
            response = requests.get(url=endpoint)

        try:
            data = response.json()
        except json.decoder.JSONDecodeError:
            data = {}
    else:
        data = {}
        if os.path.exists(endpoint):
            with open(endpoint) as fd:
                try:
                    data = json.load(fd)
                except json.JSONDecodeError:
                    data = {}

    return data


def save_dict(endpoint, data):
    if endpoint.startswith(("http:", "https:", "ftp:")):
        proxies = autodetect_proxy()

        if proxies:
            _ = requests.patch(
                url=endpoint,
                headers={"content-type": "application/json"},
                data=json.dumps(data, indent=4, sort_keys=True),
                proxies=proxies,
            )
        else:
            _ = requests.patch(
                url=endpoint,
                headers={"content-type": "application/json"},
                data=json.dumps(data, indent=4, sort_keys=True),
            )
    else:
        with open(endpoint, "w") as fd:
            json.dump(data, fd, indent=4)


PENALTY_VALUE_DICT = {
    # Traffic events that substract a set amount of points.
    TrafficEventType.COLLISION_PEDESTRIAN: 0.5,
    TrafficEventType.COLLISION_VEHICLE: 0.6,
    TrafficEventType.COLLISION_STATIC: 0.65,
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 0.7,
    TrafficEventType.STOP_INFRACTION: 0.8,
    TrafficEventType.SCENARIO_TIMEOUT: 0.7,
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: 0.7,
}
PENALTY_PERC_DICT = {
    # Traffic events that substract a varying amount of points. This is the per unit value.
    # 'increases' means that the higher the value, the higher the penalty.
    # 'decreases' means that the ideal value is 100 and the lower the value, the higher the penalty.
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: [
        0,
        "increases",
    ],  # All route traversed through outside lanes is ignored
    TrafficEventType.MIN_SPEED_INFRACTION: [0.7, "decreases"],
}

PENALTY_NAME_DICT = {
    TrafficEventType.COLLISION_STATIC: "collisions_layout",
    TrafficEventType.COLLISION_PEDESTRIAN: "collisions_pedestrian",
    TrafficEventType.COLLISION_VEHICLE: "collisions_vehicle",
    TrafficEventType.TRAFFIC_LIGHT_INFRACTION: "red_light",
    TrafficEventType.STOP_INFRACTION: "stop_infraction",
    TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION: "outside_route_lanes",
    TrafficEventType.MIN_SPEED_INFRACTION: "min_speed_infractions",
    TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE: "yield_emergency_vehicle_infractions",
    TrafficEventType.SCENARIO_TIMEOUT: "scenario_timeouts",
    TrafficEventType.ROUTE_DEVIATION: "route_dev",
    TrafficEventType.VEHICLE_BLOCKED: "vehicle_blocked",
}

# Limit the entry status to some values. Eligible should always be gotten from this table
ENTRY_STATUS_VALUES = ["Started", "Finished", "Rejected", "Crashed", "Invalid"]
ELIGIBLE_VALUES = {
    "Started": False,
    "Finished": True,
    "Rejected": False,
    "Crashed": False,
    "Invalid": False,
}

# Dictionary mapping a route failure with the 'entry status' and 'status'
FAILURE_MESSAGES = {
    "Simulation": ["Crashed", "Simulation crashed"],
    "Sensors": ["Rejected", "Agent's sensors were invalid"],
    "Agent_init": ["Started", "Agent couldn't be set up"],
    "Agent_runtime": ["Started", "Agent crashed"],
}

ROUND_DIGITS = 3
ROUND_DIGITS_SCORE = 6


class RouteRecord:
    def __init__(self):
        self.index = -1
        self.route_id = None
        self.status = "Started"
        self.num_infractions = 0
        self.infractions = {}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions[event_name] = []
        self.infractions["route_timeout"] = []

        self.scores = {"score_route": 0, "score_penalty": 0, "score_composed": 0}

        self.meta = {
            "route_length": 0,
            "duration_game": 0,
            "duration_system": 0,
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)


class GlobalRecord:
    def __init__(self):
        self.index = -1
        self.route_id = -1
        self.status = "Perfect"
        self.infractions = {}
        for event_name in PENALTY_NAME_DICT.values():
            self.infractions[event_name] = 0
        self.infractions["route_timeout"] = 0

        self.scores_mean = {"score_composed": 0, "score_route": 0, "score_penalty": 0}
        self.scores_std_dev = self.scores_mean.copy()

        self.meta = {
            "total_length": 0,
            "duration_game": 0,
            "duration_system": 0,
            "exceptions": [],
        }

    def to_json(self):
        """Return a JSON serializable object"""
        return vars(self)


class Checkpoint:
    def __init__(self):
        self.global_record = {}
        self.progress = []
        self.records = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d["global_record"] = self.global_record.to_json() if self.global_record else {}
        d["progress"] = self.progress
        d["records"] = []
        d["records"] = [x.to_json() for x in self.records if x.index != -1]  # Index -1 = Route in progress

        return d


class Results:
    def __init__(self):
        self.checkpoint = Checkpoint()
        self.entry_status = "Started"
        self.eligible = ELIGIBLE_VALUES[self.entry_status]
        self.sensors = []
        self.values = []
        self.labels = []

    def to_json(self):
        """Return a JSON serializable object"""
        d = {}
        d["_checkpoint"] = self.checkpoint.to_json()
        d["entry_status"] = self.entry_status
        d["eligible"] = self.eligible
        d["sensors"] = self.sensors
        d["values"] = self.values
        d["labels"] = self.labels

        return d


def to_route_record(record_dict):
    record = RouteRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)

    return record


def compute_route_length(route):
    route_length = 0.0
    previous_location = None

    for transform, _ in route:
        location = transform.location
        if previous_location:
            dist_vec = location - previous_location
            route_length += dist_vec.length()
        previous_location = location

    return route_length


class StatisticsManager:
    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self, endpoint, debug_endpoint):
        self._scenario = None
        self._route_length = 0
        self._total_routes = 0
        self._results = Results()
        self._endpoint = endpoint
        self._debug_endpoint = debug_endpoint

    def add_file_records(self, endpoint):
        """Reads a file and saves its records onto the statistics manager"""
        data = fetch_dict(endpoint)

        if data:
            route_records = dictor(data, "_checkpoint.records")
            if route_records:
                for record in route_records:
                    self._results.checkpoint.records.append(to_route_record(record))

    def clear_records(self):
        """Cleanes up the file"""
        if not self._endpoint.startswith(("http:", "https:", "ftp:")):
            with open(self._endpoint, "w") as fd:
                fd.truncate(0)

    def sort_records(self):
        """Sorts the route records according to their route id (This being i.e RouteScenario0_rep0)"""
        self._results.checkpoint.records.sort(
            key=lambda x: (
                int(x.route_id.split("_")[1]),
                int(x.route_id.split("_rep")[-1]),
            )
        )

        for i, record in enumerate(self._results.checkpoint.records):
            record.index = i

    def write_live_results(self, index, ego_speed, ego_control, ego_location):
        """Writes live results"""
        route_record = self._results.checkpoint.records[index]

        all_events = []
        if self._scenario:
            for node in self._scenario.get_criteria():
                all_events.extend(node.events)

        all_events.sort(key=lambda e: e.get_frame(), reverse=True)

        with open(self._debug_endpoint, "w") as f:
            f.write(
                "Route id: {}\n\n"
                "Scores:\n"
                "    Driving score:      {:.3f}\n"
                "    Route completion:   {:.3f}\n"
                "    Infraction penalty: {:.3f}\n\n"
                "    Route length:    {:.3f}\n"
                "    Game duration:   {:.3f}\n"
                "    System duration: {:.3f}\n\n"
                "Ego:\n"
                "    Throttle:           {:.3f}\n"
                "    Brake:              {:.3f}\n"
                "    Steer:              {:.3f}\n\n"
                "    Speed:           {:.3f} km/h\n\n"
                "    Location:           ({:.3f} {:.3f} {:.3f})\n\n"
                "Total infractions: {}\n"
                "Last 5 infractions:\n".format(
                    route_record.route_id,
                    route_record.scores["score_composed"],
                    route_record.scores["score_route"],
                    route_record.scores["score_penalty"],
                    route_record.meta["route_length"],
                    route_record.meta["duration_game"],
                    route_record.meta["duration_system"],
                    ego_control.throttle,
                    ego_control.brake,
                    ego_control.steer,
                    ego_speed * 3.6,
                    ego_location.x,
                    ego_location.y,
                    ego_location.z,
                    route_record.num_infractions,
                )
            )
            for e in all_events[:5]:
                # Prevent showing the ROUTE_COMPLETION event.
                event_type = e.get_type()
                if event_type == TrafficEventType.ROUTE_COMPLETION:
                    continue
                string = "    " + str(e.get_type()).replace("TrafficEventType.", "")
                if event_type in PENALTY_VALUE_DICT:
                    string += " (penalty: " + str(PENALTY_VALUE_DICT[event_type]) + ")\n"
                elif event_type in PENALTY_PERC_DICT:
                    string += " (value: " + str(round(e.get_dict()["percentage"], 3)) + "%)\n"

                f.write(string)

    def save_sensors(self, sensors):
        self._results.sensors = sensors

    def save_entry_status(self, entry_status):
        if entry_status not in ENTRY_STATUS_VALUES:
            raise ValueError("Found an invalid value for 'entry_status'")
        self._results.entry_status = entry_status
        self._results.eligible = ELIGIBLE_VALUES[entry_status]

    def save_progress(self, route_index, total_routes):
        self._results.checkpoint.progress = [route_index, total_routes]
        self._total_routes = total_routes

    def create_route_data(self, route_id, index):
        """
        Creates the basic route data.
        This is done at the beginning to ensure the data is saved, even if a crash occurs
        """
        route_record = RouteRecord()
        route_record.route_id = route_id

        # Check if we have to overwrite an element (when resuming), or create a new one
        route_records = self._results.checkpoint.records
        if index < len(route_records):
            self._results.checkpoint.records[index] = route_record
        else:
            self._results.checkpoint.records.append(route_record)

    def set_scenario(self, scenario):
        """Sets the scenario from which the statistics will be taken"""
        self._scenario = scenario
        self._route_length = round(compute_route_length(scenario.route), ROUND_DIGITS)

    def remove_scenario(self):
        """Removes the scenario"""
        self._scenario = None
        self._route_length = 0

    def compute_route_statistics(
        self,
        route_index,
        duration_time_system=-1,
        duration_time_game=-1,
        failure_message="",
    ):
        """
        Compute the current statistics by evaluating all relevant scenario criteria.
        Failure message will not be empty if an external source has stopped the simulations (i.e simulation crash).
        For the rest of the cases, it will be filled by this function depending on the criteria.
        """

        def set_infraction_message():
            infraction_name = PENALTY_NAME_DICT[event.get_type()]
            route_record.infractions[infraction_name].append(event.get_message())

        def set_score_penalty(score_penalty):
            event_value = event.get_dict()["percentage"]
            penalty_value, penalty_type = PENALTY_PERC_DICT[event.get_type()]
            if penalty_type == "decreases":
                score_penalty *= 1 - (1 - penalty_value) * (1 - event_value / 100)
            elif penalty_type == "increases":
                score_penalty *= 1 - (1 - penalty_value) * event_value / 100
            else:
                raise ValueError("Found a criteria with an unknown penalty type")
            return score_penalty

        route_record = self._results.checkpoint.records[route_index]
        route_record.index = route_index

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0
        for event_name in PENALTY_NAME_DICT.values():
            route_record.infractions[event_name] = []

        # Update the route meta
        route_record.meta["route_length"] = self._route_length
        route_record.meta["duration_game"] = round(duration_time_game, ROUND_DIGITS)
        route_record.meta["duration_system"] = round(duration_time_system, ROUND_DIGITS)

        # Update the route infractions
        if self._scenario:
            if self._scenario.timeout_node.timeout:
                route_record.infractions["route_timeout"].append("Route timeout.")
                failure_message = "Agent timed out"

            for node in self._scenario.get_criteria():
                for event in node.events:
                    # Traffic events that substract a set amount of points
                    if event.get_type() in PENALTY_VALUE_DICT:
                        score_penalty *= PENALTY_VALUE_DICT[event.get_type()]
                        set_infraction_message()

                    # Traffic events that substract a varying amount of points
                    elif event.get_type() in PENALTY_PERC_DICT:
                        score_penalty = set_score_penalty(score_penalty)
                        set_infraction_message()

                    # Traffic events that stop the simulation
                    elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                        failure_message = "Agent deviated from the route"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                        failure_message = "Agent got blocked"
                        set_infraction_message()

                    elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                        score_route = event.get_dict()["route_completed"]
                        target_reached = score_route >= 100

        # Update route scores
        route_record.scores["score_route"] = round(score_route, ROUND_DIGITS_SCORE)
        route_record.scores["score_penalty"] = round(score_penalty, ROUND_DIGITS_SCORE)
        route_record.scores["score_composed"] = round(max(score_route * score_penalty, 0.0), ROUND_DIGITS_SCORE)

        # Update result
        route_record.num_infractions = sum([len(route_record.infractions[key]) for key in route_record.infractions])

        if target_reached:
            route_record.status = "Completed" if route_record.num_infractions > 0 else "Perfect"
        else:
            route_record.status = "Failed"
            if failure_message:
                route_record.status += " - " + failure_message

        # Add the new data, or overwrite a previous result (happens when resuming the simulation)
        record_len = len(self._results.checkpoint.records)
        if route_index == record_len:
            self._results.checkpoint.records.append(route_record)
        elif route_index < record_len:
            self._results.checkpoint.records[route_index] = route_record
        else:
            raise ValueError("Not enough entries in the route record")

    def compute_global_statistics(self):
        """Computes and saves the global statistics of the routes"""

        def get_infractions_value(route_record, key):
            # Special case for the % based criteria. Extract the meters from the message. Very ugly, but it works
            if key == PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                if not route_record.infractions[key]:
                    return 0
                return float(route_record.infractions[key][0].split(" ")[8]) / 1000

            return len(route_record.infractions[key])

        global_record = GlobalRecord()
        global_result = global_record.status

        route_records = self._results.checkpoint.records

        # Calculate the score's means and result
        for route_record in route_records:
            global_record.scores_mean["score_route"] += route_record.scores["score_route"] / self._total_routes
            global_record.scores_mean["score_penalty"] += route_record.scores["score_penalty"] / self._total_routes
            global_record.scores_mean["score_composed"] += route_record.scores["score_composed"] / self._total_routes

            global_record.meta["total_length"] += route_record.meta["route_length"]
            global_record.meta["duration_game"] += route_record.meta["duration_game"]
            global_record.meta["duration_system"] += route_record.meta["duration_system"]

            # Downgrade global result if need be ('Perfect' -> 'Completed' -> 'Failed'), and record the failed routes
            route_result = "Failed" if "Failed" in route_record.status else route_record.status
            if route_result == "Failed":
                global_record.meta["exceptions"].append(
                    (route_record.route_id, route_record.index, route_record.status)
                )
                global_result = route_result
            elif global_result == "Perfect" and route_result != "Perfect":
                global_result = route_result

        for item in global_record.scores_mean:
            global_record.scores_mean[item] = round(global_record.scores_mean[item], ROUND_DIGITS_SCORE)
        global_record.status = global_result

        # Calculate the score's standard deviation
        if self._total_routes == 1:
            for key in global_record.scores_std_dev:
                global_record.scores_std_dev[key] = 0
        else:
            for route_record in route_records:
                for key in global_record.scores_std_dev:
                    diff = route_record.scores[key] - global_record.scores_mean[key]
                    global_record.scores_std_dev[key] += math.pow(diff, 2)

            for key in global_record.scores_std_dev:
                value = round(
                    math.sqrt(global_record.scores_std_dev[key] / float(self._total_routes - 1)),
                    ROUND_DIGITS,
                )
                global_record.scores_std_dev[key] = value

        # Calculate the number of infractions per km
        km_driven = 0
        for route_record in route_records:
            km_driven += route_record.meta["route_length"] / 1000 * route_record.scores["score_route"] / 100
            for key in global_record.infractions:
                global_record.infractions[key] += get_infractions_value(route_record, key)
        km_driven = max(km_driven, 0.001)

        for key in global_record.infractions:
            # Special case for the % based criteria.
            if key != PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]:
                global_record.infractions[key] /= km_driven
            global_record.infractions[key] = round(global_record.infractions[key], ROUND_DIGITS)

        # Save the global records
        self._results.checkpoint.global_record = global_record

        # Change the values and labels. These MUST HAVE A MATCHING ORDER
        self._results.values = [
            str(global_record.scores_mean["score_composed"]),
            str(global_record.scores_mean["score_route"]),
            str(global_record.scores_mean["score_penalty"]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_PEDESTRIAN]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_VEHICLE]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.COLLISION_STATIC]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.TRAFFIC_LIGHT_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.STOP_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.ROUTE_DEVIATION]]),
            str(global_record.infractions["route_timeout"]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.VEHICLE_BLOCKED]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.YIELD_TO_EMERGENCY_VEHICLE]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.SCENARIO_TIMEOUT]]),
            str(global_record.infractions[PENALTY_NAME_DICT[TrafficEventType.MIN_SPEED_INFRACTION]]),
        ]

        self._results.labels = [
            "Avg. driving score",
            "Avg. route completion",
            "Avg. infraction penalty",
            "Collisions with pedestrians",
            "Collisions with vehicles",
            "Collisions with layout",
            "Red lights infractions",
            "Stop sign infractions",
            "Off-road infractions",
            "Route deviations",
            "Route timeouts",
            "Agent blocked",
            "Yield emergency vehicles infractions",
            "Scenario timeouts",
            "Min speed infractions",
        ]

        # Change the entry status and eligible
        entry_status = "Finished"
        for route_record in route_records:
            route_status = route_record.status
            if "Simulation crashed" in route_status:
                entry_status = "Crashed"
            elif "Agent's sensors were invalid" in route_status:
                entry_status = "Rejected"

        self.save_entry_status(entry_status)

    def validate_and_write_statistics(self, sensors_initialized, crashed):
        """
        Makes sure that all the relevant data is there.
        Changes the 'entry status' to 'Invalid' if this isn't the case
        """
        error_message = ""
        if sensors_initialized and not self._results.sensors:
            error_message = "Missing 'sensors' data"

        elif not self._results.values:
            error_message = "Missing 'values' data"

        elif self._results.entry_status == "Started":
            error_message = "'entry_status' has the 'Started' value"

        else:
            global_records = self._results.checkpoint.global_record
            progress = self._results.checkpoint.progress
            route_records = self._results.checkpoint.records

            if not global_records:
                error_message = "Missing 'global_records' data"

            elif not progress:
                error_message = "Missing 'progress' data"

            elif not crashed and (progress[0] != progress[1] or progress[0] != len(route_records)):
                error_message = "'progress' data doesn't match its expected value"

            else:
                for record in route_records:
                    if record.status == "Started":
                        error_message = "Found a route record with missing data"
                        break

        if error_message:
            print("\n\033[91mThe statistics are badly formed. Setting their status to 'Invalid':")
            print(f"> {error_message}\033[0m\n")

            self.save_entry_status("Invalid")

        self.write_statistics()

    def write_statistics(self):
        """
        Writes the results into the endpoint. Meant to be used only for partial evaluations,
        use 'validate_and_write_statistics' for the final one as it only validates the data.
        """
        save_dict(self._endpoint, self._results.to_json())


def downsample_route(route, sample_factor=200):
    """
    Downsample the route by some factor, as implemented in the Leaderboard 2.0.

    Parameters
    ----------
    route: list[tuple(carla.Transform, RoadOption)]
        A list of tuples, where each tuple consists of a waypoint and a road option.
    sample_factor: int | float
        Maximum distance between two sampled waypoints.

    Returns:
    --------
    list[int]
        The list of indices for the sampled tuples.
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # At the beginning
        if prev_option is None:
            ids_to_sample.append(i)
            dist = 0

        # Lane changing
        elif curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When entering or exitting intersections
        elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i - 1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample
