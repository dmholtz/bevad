from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from typing_extensions import Self

from bevad_sim.data_interface.base_entity import BaseEntity
from bevad_sim.data_interface.routing_information import RoutingInformation


@dataclass
class EpisodeMeta(BaseEntity):
    """Holds meta data on episode level.

    Attributes:
        episode_id: Unique identifier token(s) for the episode.
        scenario_type: Describes the scenario type(s) captured in the episode,
            e.g., a CARLA scenario or Bench2Drive skill.
        region: Geographic region(s) this episode belongs to, e.g., CARLA town or
            nuScenes city district.
        weather: Describes weather aspects in the episode, e.g., CARLA weather
            parameters.
        nav_route: RoutingInformation containing waypoints in global coordinates
            describing the route to be followed.
        frame_rate: Numpy array representing the framerate (frames per second)
            of this episode's data samples.
        ego_extent: Numpy array representing the ego extents
            for this episode's data samples.
        meta: List of dictionaries (e.g. "valid_gts") containing arbitrary additional metadata.
    """

    episode_id: list[str]  # = field(default_factory=list)
    scenario_type: list[str] | None  # = field(default_factory=list)
    region: list[str] | None  # = field(default_factory=list)
    weather: list[str] | None  # = field(default_factory=list)
    nav_route: RoutingInformation | None  # = field(default_factory=lambda: np.zeros((1, 2), dtype=np.float32))
    frame_rate: np.ndarray | None  # = field(default_factory=lambda: np.zeros((1,), dtype=np.float32))
    ego_extent: np.ndarray | None
    meta: list[dict] | None  # = field(default_factory=list)

    @property
    def dimensionality(self) -> int:
        return 1

    @property
    def t_dim(self) -> None:
        """Returns the size of the time dimension. Returns none since object has no time dimension."""
        return None

    @property
    def n_dim(self) -> None:
        """Returns the size of the element dimension. Returns none since object has no element dimension."""
        return None

    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        self._check_list_dim("episode_id", 1, None, ignore_list)
        self._check_list_dim("scenario_type", 1, None, ignore_list)
        self._check_list_dim("region", 1, None, ignore_list)
        self._check_list_dim("weather", 1, None, ignore_list)
        self._check_array_dim("frame_rate", 1, None, ignore_list)
        self._check_array_dim("ego_extent", 1, (3,), ignore_list)
        self._check_list_dim("meta", 1, None, ignore_list)

    @classmethod
    def aggregated_time(cls, time: list[EpisodeMeta], use_custom_batching: list | None = None) -> EpisodeMeta:
        """Returns the most recent metadata from a list over time.

        Args:
            time: List of EpisodeMeta entries.

        Returns:
            EpisodeMeta: The last entry in the list.
        """

        return time[-1]

    def __getitem__(self, idx: int | Tuple, use_custom_slicing: List[str] | None = None) -> Self:
        ### episode meta information doesn't have time dimension except for the routing information. Remove it from idx.
        batch_only_idx = (idx[0],) if isinstance(idx, tuple) else idx
        res = super().__getitem__(batch_only_idx, ["nav_route", "ego_extent", "frame_rate"])
        if self.nav_route is not None:
            res.nav_route = self.nav_route[idx]  ## routing information has time dimension, use original idx.
        if self.ego_extent is not None:
            res.ego_extent = self.ego_extent
        if self.frame_rate is not None:
            res.frame_rate = self.frame_rate
        return res
