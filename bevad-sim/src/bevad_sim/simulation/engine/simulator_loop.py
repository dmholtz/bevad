"""Simulation Loop Module.

This module provides functions for running an environment and agent.
It supports optional evaluation with metrics,
scenario sequencing, and data writing.
"""

from __future__ import annotations

from bevad_sim.simulation.agent.base_agent import BaseAgent
from bevad_sim.simulation.engine.simulator_interface import CoreGymEnv


def simulator_loop_unbounded(env: CoreGymEnv, model: BaseAgent, num_episodes: int = 1) -> None:
    """Runs a basic simulation loop for a specified number of episodes.

    This function runs a simulation loop using the provided environment and agent
    without any scenario sequencing, metric collection, or output logging.
    It assumes that the environment is already wrapped with the necessary wrappers.

    Args:
        env: The environment to simulate.
        model: The model to control the environment.
        num_episodes: The number of episodes to run.
    """
    for _ in range(num_episodes):
        observation, info = env.reset()
        model.start_episode(env)

        while True:
            action = model.run_step(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        model.end_episode()
