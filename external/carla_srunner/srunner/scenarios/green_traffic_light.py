#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Sets the ego incoming traffic light to green. Support scenario at routes
to let the ego gather speed
"""

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import TrafficLightFreezer
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, WaitEndIntersection
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import ChangeOppositeBehavior, HandleJunctionScenario
from srunner.tools.scenario_helper import get_closest_traffic_light, get_same_dir_lanes, get_value_parameter


class PriorityAtJunction(BasicScenario):
    """
    Sets the ego incoming traffic light to green. Support scenario at routes
    to let the ego gather speed
    """

    timeout = 80  # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._tl_dict = {}

        self.timeout = timeout
        super().__init__("PriorityAtJunction",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Get the junction and traffic lights
        """
        ego_location = config.trigger_points[0].location
        self._ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_location)

        # Get the junction
        starting_wp = self._ego_wp
        ego_junction_dist = 0
        while not starting_wp.is_junction:
            starting_wps = starting_wp.next(1.0)
            if len(starting_wps) == 0:
                raise ValueError("Failed to find junction")
            starting_wp = starting_wps[0]
            ego_junction_dist += 1
        self._junction = starting_wp.get_junction()

        self._get_traffic_lights(self._junction, ego_junction_dist)

    def _get_traffic_lights(self, junction, junction_dist):
        """Get the traffic light of the junction, mapping their states"""
        tls = self._world.get_traffic_lights_in_junction(junction.id)
        if not tls:
            raise ValueError("No traffic lights found, nothing to do here")

        ego_landmark = self._ego_wp.get_landmarks_of_type(junction_dist + 1, "1000001")[0]
        ego_tl = self._world.get_traffic_light(ego_landmark)
        for tl in tls:
            self._tl_dict[tl] = carla.TrafficLightState.Green if tl.id == ego_tl.id else carla.TrafficLightState.Red

    def _create_behavior(self):
        """
        Freeze the traffic lights until the ego has exited the junction
        """
        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(WaitEndIntersection(self.ego_vehicles[0], self._junction.id))
        root.add_child(TrafficLightFreezer(self._tl_dict))
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return []

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

class PriorityAtJunctionV2(BasicScenario):
    """
    Variant of PriorityAtJunction where the ego’s light
    stays green to build up speed, then turns red just before
    the junction for a short delay, and finally back to green
    to let the ego go through.  The junction is also cleared
    beforehand so the ego can build speed.
    """

    timeout = 80  # seconds

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=80):
        """
        Setup parameters:
          - switch_traffic_light_after_distance: distance [m] before junction at which to switch to red
          - green_light_delay: how long [s] the light stays red before turning green again
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._tl_dict = {}           # priority mapping: ego-green, others-red
        self._all_red_dict = {}      # all lights red
        self._original_tl_info = {}  # to restore at end

        # read parameters (defaults: 10 m, 5 s)
        self._switch_traffic_light_after_distance = get_value_parameter(
            config, 'switch_traffic_light_after_distance', float, 10.0)
        self._green_light_delay = get_value_parameter(
            config, 'green_light_delay', float, 5.0)

        self.timeout = timeout
        super().__init__("PriorityAtJunctionV2",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Locate the junction and its traffic lights, build:
          - self._tl_dict    mapping each TL to Green if it's the ego's, else Red
          - self._all_red_dict mapping each TL to Red
        Force each light into its initial state indefinitely, storing originals.
        """
        # find ego waypoint and junction
        ego_loc = config.trigger_points[0].location
        self._ego_wp = self._map.get_waypoint(ego_loc)
        wp = self._ego_wp
        while not wp.is_junction:
            nxt = wp.next(1.0)
            if not nxt:
                raise ValueError("Failed to find junction")
            wp = nxt[0]
        self._junction = wp.get_junction()

        # collect traffic lights
        tls = self._world.get_traffic_lights_in_junction(self._junction.id)
        if not tls:
            raise ValueError("No traffic lights in junction")

        # find ego's traffic light
        ego_tl = get_closest_traffic_light(self._ego_wp, tls)

        # build mappings
        for tl in tls:
            self._tl_dict[tl] = (
                carla.TrafficLightState.Green if tl.id == ego_tl.id
                else carla.TrafficLightState.Red
            )
            self._all_red_dict[tl] = carla.TrafficLightState.Red

        # helper to freeze a TL indefinitely
        def force_set_tl(tl, state):
            self._original_tl_info[tl] = {
                "state":       tl.get_state(),
                "green_time":  tl.get_green_time(),
                "red_time":    tl.get_red_time(),
                "yellow_time": tl.get_yellow_time(),
            }
            LONG = 10000
            elapsed = tl.get_elapsed_time()
            tl.set_state(state)
            tl.set_green_time(LONG + elapsed)
            tl.set_red_time(LONG + elapsed)
            tl.set_yellow_time(LONG + elapsed)

        # apply initial freeze (priority mapping)
        for tl, state in self._tl_dict.items():
            force_set_tl(tl, state)

    def _create_behavior(self):
        """
        Sequence:
          1) (if route_mode) clear junction for ego
          2) Parallel (SUCCESS_ON_ONE):
               - WaitEndIntersection
               - TL Phase Sequence:
                   a) DriveDistance → switch_traffic_light_after_distance
                   b) TrafficLightFreezer(all-red,  duration=green_light_delay)
                   c) TrafficLightFreezer(priority mapping)
          3) (if route_mode) re-enable opposite behavior
        """
        # 1) optional clearing
        sequence = py_trees.composites.Sequence(name="PriorityAtJunctionV2")

        if self.route_mode:
            # clear the junction so ego can build speed
            sequence.add_child(HandleJunctionScenario(
                clear_junction=False,
                clear_ego_entry=True,
                remove_entries=get_same_dir_lanes(self._ego_wp),
                remove_exits=[],
                stop_entries=False,
                extend_road_exit=10.0
            ))
            sequence.add_child(ChangeOppositeBehavior(active=False))

        # 2) main parallel
        root = py_trees.composites.Parallel(
            name="PriorityAtJunctionV2Root",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        # a) end scenario when ego leaves junction
        root.add_child(WaitEndIntersection(
            self.ego_vehicles[0],
            self._junction.id
        ))

        # b) TL phase sequence
        tl_seq = py_trees.composites.Sequence(name="TL Phase Seq")
        tl_seq.add_child(DriveDistance(
            self.ego_vehicles[0],
            self._switch_traffic_light_after_distance
        ))
        tl_seq.add_child(TrafficLightFreezer(
            self._all_red_dict,
            duration=self._green_light_delay
        ))
        tl_seq.add_child(TrafficLightFreezer(self._tl_dict))

        root.add_child(tl_seq)
        sequence.add_child(root)

        # 3) restore opposite behavior if in route_mode
        if self.route_mode:
            sequence.add_child(ChangeOppositeBehavior(active=True))

        return sequence

    def _create_test_criteria(self):
        """No additional test criteria."""
        return []

    def terminate(self):
        """
        Restore all traffic lights to their original states and timings,
        then call base terminate.
        """
        for tl, info in self._original_tl_info.items():
            tl.set_state(info["state"])
            tl.set_green_time(info["green_time"])
            tl.set_red_time(info["red_time"])
            tl.set_yellow_time(info["yellow_time"])
        super().terminate()

    def __del__(self):
        """Cleanup actors."""
        self.remove_all_actors()

class SignalizedJunctionVehicleGoingStraight(PriorityAtJunctionV2):
    pass

class SignalizedJunctionTurnRightSimple(PriorityAtJunctionV2):
    pass