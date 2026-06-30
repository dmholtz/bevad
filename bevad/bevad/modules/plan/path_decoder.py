import torch
import torch.nn as nn


class WaypointDecoder(nn.Module):
    """Decode waypoint features into absolute waypoint coordinates."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, path_features: torch.Tensor) -> torch.Tensor:
        waypoint_deltas = self.layers(path_features)
        waypoints = waypoint_deltas.cumsum(dim=1)
        return waypoints
