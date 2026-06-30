from __future__ import annotations

import datetime
import pathlib
import random
from abc import ABC
from copy import deepcopy
from typing import Any

import carla
import numpy as np
from gymnasium import Wrapper, spaces
from gymnasium.core import ObservationWrapper

from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.data_interface.episode_io import EpisodeIo
from bevad_sim.data_interface.tensor_observation import TensorObservation
from bevad_sim.simulation.engine.simulator_interface import CoreGymEnv


def get_wrappers(env) -> list:
    """Retrieves the list of wrapper classes applied to a Gym environment.

    This function traverses the environment's wrapper chain and collects the classes of all wrappers
    applied to the base environment.

    Args:
        env (gym.Env): The environment to inspect. This can be a wrapped or base environment.

    Returns:
        A list of wrapper classes in the order they were applied, starting from the outermost wrapper.
    """
    wrappers = []
    while isinstance(env, Wrapper):
        wrappers.append(env.__class__)
        env = env.env
    return wrappers


class PolicyDownsampleWrapper(Wrapper):
    """Wrapper that downsamples the policy's step frequency relative to the environment's observer frequency."""

    def __init__(self, env, policy_freq: int = 1, observer_freq: int | None = 1) -> None:
        super().__init__(env)

        observer_freq = observer_freq or self.unwrapped.config["observer_freq"]
        assert observer_freq % policy_freq == 0, "observer_freq must be divisible by policy_freq"
        assert observer_freq >= policy_freq >= 1, "policy_freq cannot be more than observer_freq or less than 1"

        self.num_base_steps = observer_freq // policy_freq

    def step(self, action):
        """Runs `num_base_steps` environment steps and returns batched observations with last step's rewards."""
        for _ in range(self.num_base_steps):
            observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info


class LoggerWrapper(Wrapper, ABC):
    """A base wrapper class that provides functionalities to log CoreContainer data.

    This class is designed to be used as a wrapper around a CoreGym environment.
    It logs container data at each step and when episodes end.
    It also provides hooks for additional logging or custom behavior during
    episode start and end.
    """

    def __init__(self, env: CoreGymEnv) -> None:
        Wrapper.__init__(self, env)

    def step(self, action):
        self.log_container(action, terminated=False)
        obs, rew, ter, tru, inf = super().step(action)

        if ter or tru:
            self.log_container(obs, terminated=True)
            self.on_episode_end()

        return obs, rew, ter, tru, inf

    def reset(self, *, seed=None, options=None):
        container, info = super().reset(seed=seed, options=options)
        self.on_episode_start(container)
        return container, info

    def log_container(self, container: CoreContainer, terminated: bool = False) -> None:
        """Logs the container data.

        Args:
            container: The container object to log.
        """
        pass

    def on_episode_start(self, container: CoreContainer) -> None:
        """Called when a new episode starts."""
        pass

    def on_episode_end(self) -> None:
        """Called when an episode ends."""
        pass


class DataWriterWrapper(LoggerWrapper):
    """A Gym environment wrapper that records episode data to disk.

    This wrapper the CoreContainer at each step, during environment interactions.
    Data is written to disk when an episode terminates or is truncated. Each
    episode is stored in a timestamped subdirectory.

    Args:
        env (gym.Env): The environment to wrap.
        output_dir (pathlib.Path, optional): Base directory to save episode data. If None, data will
            be saved in timestamped directories in the current working directory.
        infer_episode_id (bool): If True, infer the directory name from the episode ID in the container.
            Defaults to False.
        ignore_first_n (int): Number of initial steps to ignore for data collection. Defaults to 0.
            This is useful for environments like CARLA where the first few steps may not contain valid data.
        downsample_by_n (int): Factor by which to downsample the data. Defaults to 1 (no downsampling).
            This is useful for reducing the amount of data collected, especially in high-frequency environments.
    """

    def __init__(
        self,
        env,
        output_dir: pathlib.Path | None = None,
        infer_episode_id: bool = False,
        ignore_first_n: int = 0,
        downsample_by_n: int = 1,
    ) -> None:
        super().__init__(env)

        self.output_dir = output_dir
        self.infer_episode_id = infer_episode_id
        self.ignore_first_n = ignore_first_n
        self.downsample_by_n = downsample_by_n
        self._data = []

    def on_episode_end(self):
        aggregated_obs = CoreContainer.aggregated_time(self._data)
        EpisodeIo.write_episode(str(self.episode_dir), aggregated_obs, config={})
        self._data = []

    def on_episode_start(self, container: CoreContainer):
        # initialize recodering
        if self.infer_episode_id:
            # pick the episode ID
            self.episode_dir = container.episode_meta.episode_id[0]
        else:
            # use a timestamp
            dt = datetime.datetime.now()
            self.episode_dir = pathlib.Path(
                f"{dt.year}-{dt.month}-{dt.day}_{dt.hour}-{dt.minute}-{dt.second}_{self.unwrapped.reset_count}"
            )
        if self.output_dir:
            self.episode_dir = self.output_dir / self.episode_dir
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self._num_calls_step = -1

        return super().on_episode_start(container)

    def log_container(self, container: CoreContainer, terminated: bool = False):
        # increment the call counter
        self._num_calls_step += 1

        # skip the first `ignore_first_n` calls to step
        if self._num_calls_step < self.ignore_first_n:
            return

        # downsample the data if required
        if self.downsample_by_n > 1 and self._num_calls_step % self.downsample_by_n != 0 and not terminated:
            return

        copied_container = deepcopy(container)

        # Save payload data of TensorObservations
        for v in copied_container.__dict__.values():
            if isinstance(v, TensorObservation):
                v.save_data(base_path=self.episode_dir)
                v.unload_data()

        self._data.append(copied_container)

    @staticmethod
    def get_episode_dir(env):
        while True:
            if isinstance(env, DataWriterWrapper):
                return env.episode_dir
            if isinstance(env, Wrapper):
                env = env.env
            else:
                break
        return None


class RandomScenarioSetter(Wrapper):
    """
    A wrapper for an environment that sets scenarios randomly from a list.
    The list should contain the scenarios configurations in the format required by the specific simulator.
    E.g. CARLASimulator expects a Path to the XML file

    Attributes:
        scenarios (list): The scenarios to be used for the environment. This value is set as the
            'scenario_config' in the options during the environment reset.
    """

    def __init__(self, env, scenarios: list[Any]):
        Wrapper.__init__(self, env)
        self.scenarios = scenarios

    def _selection_logic(self):
        return random.choice(self.scenarios)

    def reset(self, *, seed=None, options=None):
        scenario = self._selection_logic()

        if options is None:
            options = {"scenario_config": scenario}
        else:
            options["scenario_config"] = scenario

        return super().reset(seed=seed, options=options)


class SequentialScenarioSetter(RandomScenarioSetter):
    """
    A wrapper for an environment that sets scenarios sequentially from a list.
    The list should contain the scenarios configurations in the format required by the specific simulator.
    E.g. CARLASimulator expects a Path to the XML file

    Attributes:
        scenarios (list): The scenarios to be used for the environment. This value is set as the
            'scenario_config' in the options during the environment reset.
    """

    def __init__(self, env, scenarios):
        super().__init__(env, scenarios)
        self.index = 0

    def _selection_logic(self):
        config = self.scenarios[self.index]
        self.index += 1
        if self.index >= len(self.scenarios):
            self.index = 0

        return config


class RandomSeedWrapper(Wrapper):
    """A wrapper that sets a random seed for the environment."""

    def __init__(self, env, fixed_seed: int | None = 2000):
        """Initializes the wrapper with an optional seed.

        Args:
            env: The environment to wrap.
            fixed_seed (int | None): The fixed random seed to make the environment deterministic. If None, a random
            seed will be set on every reset. Defaults to 2000 (which mimics CARLA's srunner default seed).
        """

        super().__init__(env)
        self.fixed_seed = fixed_seed

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # provided seed overrides the fixed seed
            chosen_seed = seed
        elif self.fixed_seed is not None:
            # use fixed seed if available
            chosen_seed = self.fixed_seed
        else:
            # otherwise, use a random seed
            chosen_seed = random.randint(0, 2**31 - 1)

        return super().reset(seed=chosen_seed, options=options)
