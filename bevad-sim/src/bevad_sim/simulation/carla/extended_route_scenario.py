from typing import Dict, Type

import py_trees
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ScenarioTriggerer
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
from srunner.scenarios.background_activity import BackgroundBehavior
from srunner.tools.route_parser import DIST_THRESHOLD

from bevad_sim.simulation.carla.leaderboard_tools import RouteScenario


class ExtendedRouteScenario(RouteScenario):
    def __init__(
        self,
        world,
        config,
        default_background=True,
        custom_scenarios_path=None,
        rct_window_size=20,
        debug_mode=0,
        criteria_enable=True,
        enforce_behavior: bool = False,
    ):
        """Setup all relevant parameters and create scenarios along route"""

        # set these settings before calling super constructor
        self.default_background = default_background
        self.rct_window_size = rct_window_size
        self.custom_scenarios_path = custom_scenarios_path

        super().__init__(
            world,
            config,
            debug_mode,
            criteria_enable,
            enforce_behavior=enforce_behavior,
        )

    def _create_behavior(self):
        """Creates a parallel behavior that runs all of the scenarios part of the route.
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
        if self.default_background:
            behavior.add_child(BackgroundBehavior(self.ego_vehicles[0], self.route, name="BackgroundActivity"))

        behavior.add_children(scenario_behaviors)
        return behavior

    def _create_test_criteria(self):
        """Create the criteria tree. It starts with some route criteria (which are always active),
        and adds the scenario specific ones, which will only be active during their scenario
        """
        criteria = py_trees.composites.Parallel(name="Criteria", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        self.criteria_node = criteria

        # End condition
        rct = RouteCompletionTest(self.ego_vehicles[0], route=self.route)
        rct.WINDOWS_SIZE = self.rct_window_size  # important for detecting collisions at high speed
        criteria.add_child(rct)

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

    def _find_scenario_classes(self, scenarios_pkg_name="srunner.scenarios") -> Dict[str, Type]:
        "Find all scenario classes provided by srunner."

        # find the standard scenario classes
        scenario_classes = super()._find_scenario_classes()

        # if a custom scenarios path is provided, find those classes as well
        if self.custom_scenarios_path is not None:
            custom_scenario_classes = super()._find_scenario_classes(self.custom_scenarios_path)
            scenario_classes.update(custom_scenario_classes)

        return scenario_classes
