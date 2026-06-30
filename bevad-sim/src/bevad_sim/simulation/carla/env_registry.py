"""
This module registers various CARLA-based environments using Gymnasium.

Environments include base CARLA environments, those with different action representations,
and scenarios for data collection and lifecycle management.

Todo:
    * Add docstrings to each env registration
"""

from gymnasium.envs.registration import register

from bevad_sim.simulation.carla.base_wrappers import (
    CARLAMetricActionWrapper,
    ScenarioLifecycleWrapper,
)

register(
    id="CARLA/BaseEnv-v0",
    entry_point="bevad_sim.simulation.carla.simulator:CARLASimulator",
)

register(
    id="CARLA/BaseEnvMetricAct-v0",
    entry_point="bevad_sim.simulation.carla.simulator:CARLASimulator",
    additional_wrappers=(CARLAMetricActionWrapper.wrapper_spec(),),
)

register(
    id="CARLA/ScenarioEnv-v0",
    entry_point="bevad_sim.simulation.carla.simulator:CARLASimulator",
    additional_wrappers=(ScenarioLifecycleWrapper.wrapper_spec(permit_infractions=True),),
)

register(
    id="CARLA/StrictScenarioEnv-v0",
    entry_point="bevad_sim.simulation.carla.simulator:CARLASimulator",
    additional_wrappers=(ScenarioLifecycleWrapper.wrapper_spec(permit_infractions=False),),
)
