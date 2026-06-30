from copy import deepcopy

import numpy as np

from bevad_sim.data_interface.routing_information import RoutingInformation


# adapted from https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/team_code/planner.py
class RoutePlanner(object):
    def __init__(self, routing_information: RoutingInformation, min_distance, max_distance, lat_ref=42.0, lon_ref=2.0):
        self.routing_information = routing_information
        self.num_waypoints = len(self.routing_information.route_commands[0])
        assert self.num_waypoints >= 2
        self.next_navigation_goal = 1

        self.min_distance = min_distance
        self.max_distance = max_distance

    def get_observation_routing_info(self, pos: np.ndarray):
        current_ri = deepcopy(self.routing_information)

        cumulative_distance = 0.0
        farthest_in_range = -np.inf
        new_next_nav_goal = self.next_navigation_goal
        for i in range(self.next_navigation_goal, self.num_waypoints):
            if cumulative_distance > self.max_distance:
                break

            wp_next = self.routing_information.target_route[0, i]
            wp_old = self.routing_information.target_route[0, i - 1]
            cumulative_distance += ((wp_next[0] - wp_old[0]) ** 2 + (wp_next[1] - wp_old[1]) ** 2) ** 0.5
            distance = ((wp_next[0] - pos[0]) ** 2 + (wp_next[1] - pos[1]) ** 2) ** 0.5

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                new_next_nav_goal += 1

        self.next_navigation_goal = min(new_next_nav_goal, self.num_waypoints - 1)
        current_ri.navigation_goal = np.array([[self.next_navigation_goal]])
        return current_ri
