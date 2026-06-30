from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch
from gymnasium import Space

from bevad_sim.data_interface.action import Action
from bevad_sim.data_interface.base_entity import BaseEntity
from bevad_sim.data_interface.data_types import DynamicAgentType
from bevad_sim.data_interface.episode_map import EpisodeMapUtils, MapContainer
from bevad_sim.data_interface.episode_meta import EpisodeMeta
from bevad_sim.data_interface.odometry import Odometry
from bevad_sim.data_interface.routing_information import RoutingInformation
from bevad_sim.data_interface.step_meta import StepMeta
from bevad_sim.data_interface.tce import TrafficControlElements, TrafficControlElementsUtils
from bevad_sim.data_interface.tensor_observation import CameraObservation, LidarObservation, RadarObservation
from bevad_sim.data_interface.transforms.base_transform import BaseTransform
from bevad_sim.data_interface.transforms.util_transform import invert_transform
from bevad_sim.data_interface.world_state import TransformsOperations as trop
from bevad_sim.data_interface.world_state import WorldState, WorldStateUtils


@dataclass
class CoreContainer(BaseEntity):
    """Main container for all self driving-related data components.

    Attributes:
        episode_meta: Metadata about the current driving episode.
        step_meta: Metadata for individual time steps within an episode,
            containing step-specific information like timestamps and indices.
        odometry: Vehicle odometry data including position, orientation,
            velocity, and acceleration information from vehicle sensors.
        world_state: previliged data about the surrounding scene.
        tce: Traffic Control Elements including traffic signs, signals ..
        map_container: High-definition map data including lane geometry,
            road topology, and semantic information about the driving environment.
        action: Vehicle control actions including steering, acceleration,
            and planned trajectories.
        metric: Performance metrics and evaluation data.
    """

    episode_meta: EpisodeMeta
    step_meta: StepMeta
    odometry: Odometry | None
    world_state: WorldState | None
    tce: TrafficControlElements | None
    map_container: MapContainer | None
    routing_information: RoutingInformation | None
    action: Action | None

    @property
    def dimensionality(self) -> int:
        return 2

    @property
    def t_dim(self) -> int:
        """Returns the size of the time dimension."""
        return self.is_valid.shape[1]

    @property
    def n_dim(self) -> None:
        """Returns the size of the element dimension. Returns none since object has no element dimension."""
        return None

    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        pass

    def __str__(self):
        return str(self.episode_meta.episode_id)

    @property
    def camera_observations(self) -> dict[str, CameraObservation]:
        """Returns a dictionary of all Camera observations registered in this container.

        Returns:
            A dictionary mapping camera observation names to
            camera observation instances.
        """
        return {
            co.container_name: co
            for co in filter(lambda be: isinstance(be, CameraObservation), self._filter_base_entity().values())
        }

    @property
    def lidar_observations(self) -> dict[str, LidarObservation]:
        """Returns a dictionary of all Lidar observations registered in this container.

        Returns:
            A Dictionary mapping lidar observation names to lidar observation instances.

        """
        return {
            co.container_name: co
            for co in filter(lambda be: isinstance(be, LidarObservation), self._filter_base_entity().values())
        }

    @property
    def radar_observations(self) -> dict[str, RadarObservation]:
        """Returns a dictionary of all radar observations registered in this container.

        Returns:
            A Dictionary mapping radar observation names to radar observation instances.
        """
        return {
            ro.container_name: ro
            for ro in filter(lambda be: isinstance(be, RadarObservation), self._filter_base_entity().values())
        }

    def get_time_slice(self, start_timestamp_microsec: int, end_timestamp_microsec: int) -> CoreContainer:
        """Returns a new CoreContainer instance containing data from the specified time slice.

        Args:
            start_timestamp_microsec: Start timestamp in microseconds.
            end_timestamp_microsec: End timestamp in microseconds.
        Returns:
            A new CoreContainer instance with data from the specified time slice.
        """
        if self.step_meta is None or self.step_meta.timestamps is None:
            raise ValueError("step_meta or step_meta.timestamps is None")

        assert self.step_meta.timestamps.shape[0] == 1, "Only single batch CoreContainer is supported for time slicing."
        timestamps = self.step_meta.timestamps[0, :]
        assert timestamps.ndim == 1, "`timestamps` should be a 1-D array."
        if start_timestamp_microsec > end_timestamp_microsec:
            raise ValueError("start_timestamp_microsec must be less than or equal to end_timestamp_microsec")
        if start_timestamp_microsec > timestamps[-1] or end_timestamp_microsec < timestamps[0]:
            # No overlap, return empty container
            return self.create_empty()

        # Adjust the start and end timestamps to ensure they are within the data range
        adjusted_start_timestamp = max(start_timestamp_microsec, timestamps[0].item())
        adjusted_end_timestamp = min(end_timestamp_microsec, timestamps[-1].item())

        # Find the indices of the adjusted start and end timestamps
        try:
            start_index = next(i for i, ts in enumerate(timestamps) if ts >= adjusted_start_timestamp)
            end_index = next(i for i, ts in enumerate(timestamps) if ts >= adjusted_end_timestamp)
        except StopIteration:
            # If indices cannot be found, return empty container
            return self.create_empty()

        return self[:, start_index : end_index + 1]


class CoreContainerSpace(Space):
    """A custom observation/action space for Gym environments representing a space of CoreContainer objects."""

    def __init__(self):
        """Initializes the Space."""
        super().__init__()

    def sample(self, mask: Any | None = None, probability: Any | None = None) -> CoreContainer:
        """Samples a random CoreContainer from the space.

        Returns:
            An empty CoreContainer object.
        """
        # TODO(acaneva): check if empty container is OK
        return CoreContainer.create_empty()

    def contains(self, x: Any) -> bool:
        """Checks if a given object is contained within the space.

        Args:
            x: The object to check.

        Returns:
            True if the object is an instance of CoreContainer, False otherwise.
        """
        return isinstance(x, CoreContainer)


class CoreContainerAggregator:
    """Maintains a fixed-size buffer of CoreContainer samples for time aggregation."""

    def __init__(self, num_hist_steps: int = 10):
        """Initializes the aggregator.

        Args:
            num_hist_steps: Maximum number of samples to retain.
        """
        self.num_hist_steps = num_hist_steps
        self.data: deque[CoreContainer] = deque()

    def add_sample(self, sample: CoreContainer):
        """Adds a new sample, discarding the oldest if over limit.

        Args:
            sample: A CoreContainer sample to add.
        """

        self.data.append(sample)
        while len(self.data) > self.num_hist_steps:
            self.data.popleft()

    def get_time_aggregated(self) -> CoreContainer:
        """Returns time-aggregated CoreContainer over stored samples.

        Returns:
            A CoreContainer instance representing the aggregation.
        """
        return CoreContainer.aggregated_time(list(self.data))

    def is_full(self) -> bool:
        """Checks if the buffer is full.

        Returns:
            True if the number of samples >= num_hist_steps.
        """
        return len(self.data) >= self.num_hist_steps

    def __len__(self) -> int:
        """Returns the number of stored samples.

        Returns:
            The current size of the internal buffer.
        """
        return len(self.data)


# TODO: If possible make these into transform classes
# TODO : should not these be part of worldstate entity ?
#
class CoreContainerUtils:
    """Utility functions for querying and manipulating CoreContainer data."""

    @staticmethod
    def get_idx_by_id(obs: CoreContainer, b: int, t: int, agent_id: int) -> int:
        """Returns agent index for a given ID at a specific batch/time.

        Args:
            obs: The CoreContainer instance.
            b: Batch index.
            t: Time index.
            agent_id: ID of the agent.

        Returns:
            Index of the agent in the data array.
        """
        # mypy: obs.world_state could be None, fallback to error or default
        if obs.world_state is None:
            raise AttributeError("obs.world_state is None")
        return np.where(obs.world_state.track_id[b, t] == agent_id)[0][0]

    @staticmethod
    def get_dynamics_by_id(obs: CoreContainer, b: int, t: int, agent_id: int) -> np.ndarray | torch.Tensor:
        """Fetches dynamics of a specific agent by ID.

        Args:
            obs: The CoreContainer instance.
            b: Batch index.
            t: Time index.
            agent_id: ID of the agent.

        Returns:
            Agent's dynamics array.
        """
        assert obs.world_state is not None and obs.world_state.dynamics is not None

        if obs.world_state is None:
            raise AttributeError("obs.world_state is None")
        agent_idx = CoreContainerUtils.get_idx_by_id(obs, b, t, agent_id)
        return obs.world_state.dynamics[b, t, agent_idx]

    @staticmethod
    def auto_select_ego(episode: CoreContainer, min_vel: float = 3.0) -> list[int]:
        """Selects agents that qualify as ego candidates.

        Args:
            episode: A single-batch CoreContainer episode.
            min_vel: Minimum velocity threshold.

        Returns:
            Track IDs of selected ego candidates.
        """
        assert episode.shape[0] == 1
        assert episode.world_state is not None
        assert episode.world_state.category is not None and episode.world_state.dynamics is not None

        res_agents = []
        num_agents = episode.world_state.track_id.shape[2]
        for i in range(num_agents):
            if episode.world_state.category[0, 0, i] != DynamicAgentType.VEHICLE:
                continue
            if np.min(episode.world_state.is_valid[0, :, i]) < 1.0:
                continue
            if np.max(episode.world_state.dynamics[0, :, i, 0]) < min_vel:
                continue
            res_agents.append(int(episode.world_state.track_id[0, 0, i]))
        return res_agents

    @staticmethod
    def auto_select_ego_and_set(episode: CoreContainer, min_vel: float = 3.0):
        """Automatically selects and sets ego ID in the episode.

        Args:
            episode: A single-batch CoreContainer episode.
            min_vel: Minimum velocity threshold.
        """

        assert episode.shape[0] == 0, "only support single-batch because auto_select_ego() supports single-batch only"
        assert episode.world_state is not None
        episode.world_state.ego_id = np.array(CoreContainerUtils.auto_select_ego(episode, min_vel=min_vel)[0])

    @staticmethod
    def get_state_2d_id(obs: CoreContainer, b: int, t: int, agent_id: int) -> tuple[float, float, float]:
        """Gets 2D pose (x, y, yaw) of a given agent ID.

        Args:
            obs: The CoreContainer instance.
            b: Batch index.
            t: Time index.
            agent_id: ID of the agent.

        Returns:
            A tuple (x, y, yaw), representing the 2d pose.
        """
        assert obs.world_state is not None
        assert obs.world_state.transform is not None

        if obs.world_state is None:
            raise AttributeError("obs.world_state is None")
        agent_idx = CoreContainerUtils.get_idx_by_id(obs, b, t, agent_id)
        m = obs.world_state.transform[b, t, agent_idx]
        x, y, yaw = trop.get_xyyaw_from_transforms(m)
        return x.item(), y.item(), yaw.item()

    @staticmethod
    def get_ego_state_2d(obs: CoreContainer, b: int, t: int) -> tuple[float, float, float]:
        """Returns 2D pose (x, y, yaw) of the ego agent.

        Args:
            obs (CoreContainer): The CoreContainer instance.
            b (int): Batch index.
            t (int): Time index.

        Returns:
            A tuple (x, y, yaw), representing the pose of ego.
        """

        if obs.world_state is None:
            raise AttributeError("obs.world_state is None")
        if obs.world_state.ego_id is None:
            raise AttributeError("obs.world_state.ego_id is None")
        return CoreContainerUtils.get_state_2d_id(obs, b, t, obs.world_state.ego_id[b])
