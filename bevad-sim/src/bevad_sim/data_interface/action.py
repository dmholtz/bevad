from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch

from bevad_sim.data_interface.base_entity import BaseEntity


@dataclass
class Action(BaseEntity):
    """Action entity representing vehicle control commands and trajectory data.

    This dataclass encapsulates different representations of vehicle actions including
    metric-based control inputs, planned trajectories, and simulator-specific control
    objects.

    The class supports multiple action representations:
    - Metric control: Standardized acceleration and steering commands
    - Metric trajectory: Planned waypoint sequences in metric coordinates
    - Simulator control: Simulator-specific control objects (e.g., CARLA VehicleControl)

    All tensor attributes follow the batch-time convention where B represents batch
    size and T represents time steps. The trajectory additionally includes N waypoints
    per time step.

    Attributes:
        metric_control: Tensor containing acceleration and steering commands.
            Shape: (B, T, 2) where the last dimension represents:
            - Index 0: Acceleration in m/s²
            - Index 1: Steering angle in radians
            Can be torch.Tensor, numpy.ndarray, or None if not available.

        metric_trajectory: Tensor containing planned trajectory waypoints.
            Shape: (B, T, N, 2) where:
            - B: Batch size
            - T: Time steps
            - N: Number of waypoints per time step
            - Last dimension: (x, y) coordinates in meters
            Can be torch.Tensor, numpy.ndarray, or None if not available.

        simulator_control: Nested list containing simulator-specific control objects.
            Structure: List[List[Any]] with shape-equivalent (B, T, 1)
            - Outer list: Batch dimension
            - Inner list: Time dimension
            - Elements: Simulator-specific control objects (e.g., CARLA VehicleControl)
            None if simulator-specific controls are not needed.
    """

    metric_control: torch.Tensor | np.ndarray | None  # B, T, 2 (acc m/s2, steering rad)
    metric_trajectory: torch.Tensor | np.ndarray | None  # B, T, N, 2 (x, y)
    metric_trajectory_timestamps: torch.Tensor | np.ndarray | None  # B, T, N
    simulator_control: list[list[Any]] | None  # B, T, 1 (simulator specific control e.g. Carla.VehicleControl)

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
        self._check_array_dim("metric_trajectory", 2, (2,), ignore_list)
        self._check_array_dim("metric_control", 2, (2,), ignore_list)
        self._check_array_dim("metric_trajectory_timestamps", 2, (1,), ignore_list)
        self._check_list_dim("simulator_control", 2, None, ignore_list)
