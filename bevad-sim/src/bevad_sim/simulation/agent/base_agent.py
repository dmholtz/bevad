from __future__ import annotations

from typing import Any, Callable, Optional

from bevad_sim.data_interface.configurator import Configurator
from bevad_sim.data_interface.core_container import CoreContainer


class BaseAgent(Configurator):
    """Base class for creating agents that interact with E2ECoreGym environments."""

    def __init__(
        self, obs_transform: Callable | None = None, act_transform: Callable | None = None, config: dict | None = None
    ) -> None:
        """Initialize the BaseAgent with transforms and configuration.

        Args:
            obs_transform: Optional callable to transform observations before processing
            act_transform: Optional callable to transform actions before execution
            config: Optional configuration dictionary to override default settings
        """

        self.config = self.default_config()
        self.configure(config)
        self._configure_logger(verbose=self.config["verbose"])

        self.obs_transform = obs_transform
        self.act_transform = act_transform

    @classmethod
    def default_config(cls):
        """Default environment configuration.

        Can be overloaded in environment implementations,
        or by calling configure().

        Returns
        -------
            dict: A configuration dictionary.

        """
        config = {"verbose": 0}
        return config

    def run_step(self, container: CoreContainer) -> CoreContainer:
        """Execute a single processing step for the agent.

        The standard processing pipeline:
        1. Apply observation transformation
        2. Run model with transformed observations
        3. Apply action transformation
        4. Update container with generated action

        Args:
            container: Input data container from the environment

        Returns:
            CoreContainer: Updated container with agent's action
        """
        obs = self.obs_transform(container)
        act = self.run_model(obs)
        container_with_action = self.act_transform(act)
        container.action = container_with_action.action
        return container

    def run_model(self, data: CoreContainer) -> CoreContainer:
        """Process input data to generate actions.

        Note:
            This method should be overridden in subclasses to implement specific
            model logic. The base implementation acts as a pass-through.

        Args:
            data: Transformed observation data from the environment

        Returns:
            Any: Processed output (typically action data)
        """
        return data

    def end_episode(self):
        """
        End the current episode.
        This method is intended to be overridden by subclasses to perform any
        necessary cleanup or finalization at the end of an episode.
        If the agent creates any results (such as some video), the result is expected to be returned here.
        """

        return None

    def start_episode(self, env: Optional[Any] = None):
        """
        Starts a new episode for the agent.
        This method should be called at the beginning of each episode to initialize
        any necessary parameters or states for the agent and to reset the agent in between runs in the same environment.
        """

        pass
