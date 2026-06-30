import os
from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.models import HEADS, build_head
from mmcv.models.bricks.transformer import build_attention
from mmcv.utils import ConfigDict

from bevad.modules.plan.path_decoder import WaypointDecoder
from bevad.modules.util.diffusion import modulate


@HEADS.register_module()
class DisentangledPlanningDecoderLayer(nn.Module):
    def __init__(
        self,
        bev_range: list[float],
        d_model: int,
        d_bev: int | None,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        deformable_attn: ConfigDict | None,
        query_lengths: tuple[int, int],
    ):
        super().__init__()

        self.LOG_ATTENTION_WEIGHTS = os.environ.get("LOG_ATTENTION_WEIGHTS", "0") == "1"

        # config
        self.bev_range = bev_range
        self.query_lengths = query_lengths

        # self-attention
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # cross-attention
        self.norm_cross_attn = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # deformable cross-attention
        self.with_deformable_attn = deformable_attn is not None
        if self.with_deformable_attn:
            self.norm_deform_attn = nn.LayerNorm(d_model)
            self.deformable_attn = build_attention(deformable_attn)
            self.bev_to_model_proj = nn.Linear(d_bev, d_model)
            self.model_to_bev_proj = nn.Linear(d_model, d_bev)
            self.reference_point_decoder = WaypointDecoder(
                input_dim=d_bev, hidden_dim=d_bev // 2, output_dim=2
            )

        # feedforward network
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # adaLN modulation
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )

    def forward(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        bev: torch.Tensor | None,
        cond: torch.Tensor,
        q_padding_mask: torch.Tensor | None,
        kv_padding_mask: torch.Tensor | None,
        log_attention_weights: bool = False,
    ):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_ln_modulation(cond.squeeze(1)).chunk(9, dim=-1)

        q = self._forward_self_attn(q, q_padding_mask, shift_sa, scale_sa, gate_sa)
        q, cross_attn_values = self._forward_cross_attn(
            q,
            kv,
            kv_padding_mask,
            shift_ca,
            scale_ca,
            gate_ca,
            log_attention_weights=log_attention_weights,
        )

        if self.with_deformable_attn:
            assert bev is not None

            # split into path and trajectory queries
            q_path, q_traj = torch.split(q, self.query_lengths, dim=1)

            q_path, intermediate_path = self._forward_deformable_bev_attn(q_path, bev)

            # combine path and trajectory queries
            q = torch.cat([q_path, q_traj], dim=1)
        else:
            intermediate_path = None

        q = self._forward_ffn(q, shift_mlp, scale_mlp, gate_mlp)

        return q, dict(
            intermediate_path=intermediate_path, cross_attn_values=cross_attn_values
        )

    def _forward_self_attn(
        self,
        query: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
    ):
        mod_query = modulate(self.norm_self_attn(query), shift, scale)
        query = (
            query
            + gate.unsqueeze(1)
            * self.self_attn(
                mod_query,
                mod_query,
                mod_query,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        )
        return query

    def _forward_cross_attn(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        log_attention_weights: bool,
    ):
        mod_query = modulate(self.norm_cross_attn(query), shift, scale)
        attn, attn_values = self.cross_attn(
            mod_query,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=self.LOG_ATTENTION_WEIGHTS or log_attention_weights,
        )
        query = query + gate.unsqueeze(1) * attn
        return query, attn_values

    def _forward_deformable_bev_attn(self, query: torch.Tensor, bev: torch.Tensor):
        hw = bev.size(1)
        h = w = int(hw**0.5)

        residual = query
        query = self.norm_deform_attn(query)
        query = self.model_to_bev_proj(query)

        intermediate_path, reference_points = self._get_reference_points(query.detach())
        spatial_shapes = torch.tensor([[h, w]], device=bev.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=bev.device, dtype=torch.long)
        query = self.deformable_attn(
            query,
            value=bev,
            reference_points=reference_points[:, :, None, :].detach(),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.bev_to_model_proj(query)
        query = query + residual

        return query, intermediate_path

    def _forward_ffn(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
    ):
        mod_x = modulate(self.norm_ffn(x), shift, scale)
        x = x + gate.unsqueeze(1) * self.ffn(mod_x)
        return x

    def _get_reference_points(self, path_query: torch.Tensor):
        # decode
        path = self.reference_point_decoder(path_query)

        # flip coordinates for grid sampling with deformable attention
        path_lidar = torch.zeros_like(path)
        path_lidar[..., 0] = -path[..., 1]
        path_lidar[..., 1] = path[..., 0]

        # normalize to [0, 1] considering the BEV range
        reference_points = (path_lidar - self.bev_range[0]) / (
            self.bev_range[3] - self.bev_range[0]
        )

        return path, reference_points


@HEADS.register_module()
class DisentangledPlanningDecoder(nn.Module):
    def __init__(self, num_layers: int, decoder_layer: ConfigDict):
        super().__init__()

        layers = []
        layer_cfgs = [deepcopy(decoder_layer) for _ in range(num_layers)]
        for i, layer_cfg in enumerate(layer_cfgs):
            if i < num_layers - 1:
                # first n layers w/o deformable attention
                layer_cfg["deformable_attn"] = None
                layers.append(build_head(layer_cfg))
            else:
                # last layer with deformable attention (if configured)
                layers.append(build_head(layer_cfg))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        bev: torch.Tensor | None,
        cond: torch.Tensor,
        q_padding_mask: torch.Tensor | None,
        kv_padding_mask: torch.Tensor | None,
        # optional for loss on intermediate paths
        gt_path: torch.Tensor | None = None,
        gt_path_mask: torch.Tensor | None = None,
        log_attention_weights: bool = False,
    ):
        losses = {}
        attn_weights = []
        for i, layer in enumerate(self.layers):
            q, artifacts = layer(
                q=q,
                kv=kv,
                bev=bev,
                cond=cond,
                q_padding_mask=q_padding_mask,
                kv_padding_mask=kv_padding_mask,
                log_attention_weights=log_attention_weights,
            )

            if (
                intermediate_path := artifacts.get("intermediate_path")
            ) is not None and gt_path is not None:
                loss = (
                    nn.functional.smooth_l1_loss(
                        intermediate_path[gt_path_mask],
                        gt_path[gt_path_mask],
                        reduction="none",
                    )
                    .sum(-1)
                    .mean()
                )
                losses[f"d{i}.loss_path"] = loss
            if (cross_attn_values := artifacts.get("cross_attn_values")) is not None:
                attn_weights.append(cross_attn_values)

        if len(attn_weights):
            attn_weights = torch.stack(attn_weights, dim=0)
        else:
            attn_weights = None

        return dict(q=q, attn_weights=attn_weights), losses
