from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import gymnasium as gym

from bevad_sim.data_interface.configurator import read_dict
from bevad_sim.data_interface.episode_io import EpisodeIo
from bevad_sim.simulation.carla.base_wrappers import ScenarioLifecycleWrapper
from bevad_sim.simulation.engine.gym_wrappers import DataWriterWrapper, RandomSeedWrapper, SequentialScenarioSetter
from bevad_sim.simulation.engine.simulator_loop import simulator_loop_unbounded


def build_single_scenario_environment(
    route_file: Path,
    sensor_config: Path | dict[str, Any],
    permit_infractions: bool = True,
    timeout: float | None = None,
    output_dir: Path | None = None,
    downsample_by_n: int = 1,
    fixed_seed: int | None = None,
):
    """Builds a CARLA environment for simulation a single scenario, e.g., in data collection or closed-loop evaluation.

    Args:
        route_file (Path): Path to the XML file describing the route.
        sensor_config (Path | dict[str, Any]): Path to the JSON file describing the sensor configuration or a dictionary with sensor configuration.
        permit_infractions (bool): Whether to continue the simulation if infractions occur.
        timeout (float | None): Maximum time for the episode in seconds. If None, no timeout is applied.
        output_dir (Path | None): Directory to write data to. If None, no data is written.
        downsample_by_n (int): Factor by which to downsample the data. Defaults to 1 (no downsampling).
        fixed_seed (int | None): Fixed seed for the environment. If None, a random seed is used on every reset.

    Returns:
        gym.Env: The configured CARLA environment.
    """
    if not isinstance(sensor_config, dict):
        sensor_config = read_dict(sensor_config)

    sim_config = {
        "default_background": True,
        "verbose": 1,
        "observer_config": {"route_downsample_dist": 200},  # mimic leaderboard 2.0 setting
        "enforce_behavior": True,  # simulation fails with exception if any scenario cannot be initialized
        "agent_config": sensor_config,
    }

    if timeout is None or timeout < 0:
        max_steps = -1
    else:
        max_steps = int(timeout * 20) + 1

    env = gym.make(
        "bevad_sim.simulation.carla.env_registry:CARLA/BaseEnv-v0", max_episode_steps=max_steps, config=sim_config
    )
    env = SequentialScenarioSetter(env, [route_file])
    env = ScenarioLifecycleWrapper(env, permit_infractions=permit_infractions)
    env = RandomSeedWrapper(env, fixed_seed=fixed_seed)

    if output_dir is not None:
        env = DataWriterWrapper(
            env, output_dir=output_dir, infer_episode_id=True, ignore_first_n=8, downsample_by_n=downsample_by_n
        )

    return env


def evaluate_single_scenario(
    route_file: Path,
    sensor_config: Path | dict[str, Any],
    agent_module_name: str,
    agent_config: dict[str, Any],
    permit_infractions: bool = True,
    timeout: float | None = None,
    output_dir: Path | None = None,
    downsample_by_n: int = 1,
    fixed_seed: int | None = None,
) -> Path | None:
    """Run an agent on a single scenario in a CARLA environment.

    Args:
        route_file (Path): The path to the XML file describing the route.
        sensor_config (Path | dict[str, Any]): The path to the JSON file describing the sensor configuration or a dictionary with sensor configuration.
        agent_module_name (str): The import path / name of the agent module. The module must contain a single `get_class()`
            function, which calls the constructer of the agent that shall be created. Example:
            `bevad_sim/simulators/carla/pdm_lite_expert/pdm_lite`
        agent_config (dict): The key-value based configuration of the agent (model-specific).
        permit_infractions (bool): Whether to continue the simulation if infractions occur.
        timeout (float | None): Maximum time for the episode in seconds. If None, no timeout is applied.
        output_dir (Path | None): Directory to write data to. If None, no data is written.
        downsample_by_n (int): Factor by which to downsample the data. Defaults to 1 (no downsampling).
        fixed_seed (int | None): Fixed seed for the environment. If None, a random seed is used on every reset.

    Returns:
        Path | None: The directory where the episode data was written, or None if no data was written.

    Raises:
        RuntimeError: If the simulation fails.
    """
    # build the environment
    env = build_single_scenario_environment(
        route_file=route_file,
        sensor_config=sensor_config,
        permit_infractions=permit_infractions,
        timeout=timeout,
        output_dir=output_dir,
        downsample_by_n=downsample_by_n,
        fixed_seed=fixed_seed,
    )

    # build the agent
    agent_mod = importlib.import_module(agent_module_name)
    agent = agent_mod.get_class()(config=agent_config)

    # run the simulator loop
    simulator_loop_unbounded(env, agent, num_episodes=1)

    # get the episode directory if data was written
    episode_dir = DataWriterWrapper.get_episode_dir(env)

    # we do not close the environment, it somehow crashes the flyte workflow
    # env.close()

    return episode_dir


def check_success(episode_dir: Path) -> bool:
    episode = EpisodeIo.read_episode(str(episode_dir), load_payload=False)

    # check leaderboard metrics
    lb_metrics = episode.step_meta.info[0][-1]["lb_metrics"]
    if lb_metrics["score_route"] < 99:
        print(f"Episode {episode_dir} failed with score {lb_metrics['score_route']}")
        return False

    if len(lb_metrics["collisions_layout"]) > 0:
        print(f"Episode {episode_dir} failed due to layout collisions: {lb_metrics['collisions_layout']}")
        return False

    if len(lb_metrics["collisions_pedestrian"]) > 0:
        print(f"Episode {episode_dir} failed due to pedestrian collisions: {lb_metrics['collisions_pedestrian']}")
        return False

    if len(lb_metrics["collisions_vehicle"]) > 0:
        print(f"Episode {episode_dir} failed due to vehicle collisions: {lb_metrics['collisions_vehicle']}")
        return False

    if len(lb_metrics["red_light"]) > 0:
        print(f"Episode {episode_dir} failed due to red light infractions: {lb_metrics['red_light']}")
        return False

    if len(lb_metrics["stop_infraction"]) > 0:
        print(f"Episode {episode_dir} failed due to stop sign infractions: {lb_metrics['stop_infraction']}")
        return False

    if len(lb_metrics["yield_emergency_vehicle_infractions"]) > 0:
        print(
            f"Episode {episode_dir} failed due to yield emergency vehicle infractions: {lb_metrics['yield_emergency_vehicle_infractions']}"
        )
        return False

    if len(lb_metrics["outside_route_lanes"]) > 0:
        print(
            f"Episode {episode_dir} failed due to outside route lanes infractions: {lb_metrics['outside_route_lanes']}"
        )
        return False

    return True
