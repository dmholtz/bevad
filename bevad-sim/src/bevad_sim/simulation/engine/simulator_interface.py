from abc import ABC

import gymnasium as gym

from bevad_sim.data_interface.core_container import CoreContainerSpace


class CoreGymEnv(gym.Env, ABC):
    """Abstract base class for end-to-end Gym environments.

    This class ensures that all environments use a fixed observation space: `CoreContainerSpace`.
    It also prevents direct modification of `observation_space` unless the environment is wrapped
    by a Gym wrapper.

    it enforce similar a constraint to the action space which is: `CoreContainerSpace`.

    Attributes:
        _observation_space (CoreContainerSpace): The fixed observation space of the environment.
        _action_space (CoreContainerSpace): The action space of the environment.
    """

    def __init__(self, **kwargs):
        """Initializes the environment and sets a default observation space.

        This ensures that every subclass has a valid `observation_space` without requiring
        explicit assignment in the subclass constructor.
        """
        self._observation_space = CoreContainerSpace()  # Default value
        self._action_space = CoreContainerSpace()

        super().__init__(**kwargs)

    @property
    def observation_space(self):
        """Gets the observation space of the environment.

        Returns:
            CoreContainerSpace: The observation space of the environment.
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space):
        """Sets the observation space only if the environment is wrapped.

        This prevents subclasses from modifying `observation_space` directly while allowing
        Gym wrappers to adjust it dynamically.

        Args:
            space (CoreContainerSpace): The new observation space.
        """
        if self._is_wrapped():
            self._observation_space = space  # Allow modification in wrappers

    @property
    def action_space(self):
        """Gets the action space of the environment.

        Returns:
            CoreContainerSpace: The action space of the environment.
        """
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        """Sets the action space only if the environment is wrapped.

        This prevents subclasses from modifying `action_space` directly while allowing
        Gym wrappers to adjust it dynamically.

        Args:
            space (CoreContainerSpace): The new action space.
        """
        if self._is_wrapped():
            self._action_space = space  # Allow modification in wrappers

    def _is_wrapped(self):
        """Checks if the environment is wrapped by a Gym wrapper.

        This method traverses the `.env` attributes of the environment to determine if
        it is part of a wrapper chain.

        Returns:
            bool: True if the environment is wrapped, False otherwise.
        """
        env = self
        while hasattr(env, "env"):  # Traverse through nested environments
            env = env.env
            if isinstance(env, gym.Env):  # Found a wrapped base environment
                return True
        return False
