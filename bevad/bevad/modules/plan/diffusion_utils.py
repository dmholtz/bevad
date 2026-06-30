import torch
import math
import torch.nn as nn


def normalize_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    fut_x = trajectory[..., 0:1]
    fut_y = trajectory[..., 1:2]
    fut_sin_cos = trajectory[..., 2:4]

    fut_x = fut_x / 25 - 1
    fut_y = fut_y / 25
    return torch.cat([fut_x, fut_y, fut_sin_cos], dim=-1)


def denormalize_trajectory(norm_trajectory: torch.Tensor) -> torch.Tensor:
    fut_x = norm_trajectory[..., 0:1]
    fut_y = norm_trajectory[..., 1:2]
    fut_sin_cos = norm_trajectory[..., 2:4]

    fut_x = (fut_x + 1) * 25
    fut_y = fut_y * 25
    return torch.cat([fut_x, fut_y, fut_sin_cos], dim=-1)


def trajectory_to_distance(trajectory: torch.Tensor) -> torch.Tensor:
    traj = torch.cat(
        [torch.zeros_like(trajectory[:, 0:1, 0:2]), trajectory[..., 0:2]], dim=1
    )
    dist = traj[:, 1:] - traj[:, :-1]
    dist = torch.square(dist).sum(dim=-1).sqrt()
    dist = torch.cumsum(dist, dim=1)
    return dist


def normalize_trajectory_dist(dist: torch.Tensor) -> torch.Tensor:
    return dist / 20 - 1


def denormalize_trajectory_dist(norm_dist: torch.Tensor) -> torch.Tensor:
    return (norm_dist + 1) * 20


def normalize_path(path: torch.Tensor) -> torch.Tensor:
    fut_x = path[..., 0:1] / 25 - 1
    fut_y = path[..., 1:2] / 25

    return torch.cat([fut_x, fut_y], dim=-1)


def denormalize_path(norm_path: torch.Tensor) -> torch.Tensor:
    fut_x = (norm_path[..., 0:1] + 1) * 25
    fut_y = norm_path[..., 1:2] * 25

    return torch.cat([fut_x, fut_y], dim=-1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

