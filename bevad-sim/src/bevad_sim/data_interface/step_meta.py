from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import torch

from bevad_sim.data_interface.base_entity import BaseEntity


@dataclass
class StepMeta(BaseEntity):
    """
    Holds meta data on step-level for reinforcement learning environments.

    Attributes:
        frame_ids: List of frame IDs
        timestamps: List of timestamps. TODO: units need to be discussed
        reward: Contains the reward for the last transition
        terminated: True if the episode is terminated
        truncated: True if the episode is truncated
        info: Additional info dictionaries from the environment for each step.

    Properties:
        shape (tuple): Shape of the first non-None attribute among is_step_valid, frame_ids, timestamps, reward, terminated, or truncated.
    """

    frame_ids: np.ndarray | torch.Tensor
    timestamps: np.ndarray | torch.Tensor
    reward: np.ndarray | torch.Tensor | None
    terminated: np.ndarray | torch.Tensor | None
    truncated: np.ndarray | torch.Tensor | None
    info: list[list[dict]] | None

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
        self._check_array_dim("frame_ids", 2, None, ignore_list)
        self._check_array_dim("timestamps", 2, None, ignore_list)
        self._check_array_dim("reward", 2, None, ignore_list)
        self._check_array_dim("terminated", 2, None, ignore_list)
        self._check_array_dim("truncated", 2, None, ignore_list)
        self._check_list_dim("info", 2, None, ignore_list)
