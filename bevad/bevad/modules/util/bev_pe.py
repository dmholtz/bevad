import torch
import torch.nn as nn

import math


def sinusoidal_encoding(x: torch.Tensor, dim: int, omega_max: float = 1e4):
    assert dim % 2 == 0
    d = dim // 2

    log_omega_max = torch.log(x.new_tensor(omega_max))
    omegas = torch.exp(
        -log_omega_max * torch.arange(0, d, 1, dtype=x.dtype, device=x.device) / d
    )

    sin_encoding = torch.sin(omegas * x.unsqueeze(-1))
    cos_encoding = torch.cos(omegas * x.unsqueeze(-1))
    return torch.cat((sin_encoding, cos_encoding), dim=-1)


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


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(
        self,
        *,
        len_x: int,
        len_y: int,
        d_model: int,
    ):
        super().__init__()

        assert d_model % 4 == 0, "d_model must be divisible by 4"

        pe = self._precompute_grid_encoding(len_x, len_y, d_model)
        self.register_buffer("pe", pe)

    def _precompute_grid_encoding(self, len_x, len_y, channels):
        # number of channels for x and y dimension
        channels_half = channels // 2

        pos_x = torch.arange(0, len_x)
        pe_x = sinusoidal_encoding(pos_x, channels_half)
        pos_y = torch.arange(0, len_y)
        pe_y = sinusoidal_encoding(pos_y, channels_half)

        # combine x and y positional encodings into grid
        pe_x = pe_x.unsqueeze(1).expand(-1, len_y, -1)
        pe_y = pe_y.unsqueeze(0).expand(len_x, -1, -1)
        pe = torch.cat((pe_x, pe_y), dim=-1)

        # flatten the grid to a single dimension
        return pe.flatten(0, 1).contiguous()

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        batch_size = bev.size(0)
        pe = self.pe.unsqueeze(0).expand(batch_size, -1, -1)
        return pe
