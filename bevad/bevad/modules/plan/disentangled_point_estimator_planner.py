import math

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from mmcv.models import HEADS, build_head
from mmcv.utils import ConfigDict

from bevad.modules.util.bev_pe import (
    SinusoidalPosEmb,
    SinusoidalPositionalEncoding2D,
)
from bevad.modules.plan.diffusion_utils import (
    normalize_path,
    normalize_trajectory,
    normalize_trajectory_dist,
    trajectory_to_distance,
)
from bevad.modules.plan.path_decoder import WaypointDecoder
from bevad.modules.plan.rollout import BezierTrajectoryRollout


@HEADS.register_module()
class DisentangledPointEstimatorPlanner(nn.Module):
    def __init__(
        self,
        # representation
        num_commands: int,
        num_planning_steps: int,
        plan_with_speed: bool,
        num_bev_waypoints: int,
        # transformer
        disentangled_decoder: ConfigDict,
        # tokenizer
        d_bev: int,
        d_model: int,
        bev_pooling: int | None,
        bev_unshuffling: int | None,
        bev_size: int,
        loss_weight: float,
        cfg_p_uncond: float,
        **kwargs,
    ):
        super().__init__()

        # config
        self.num_planning_steps = num_planning_steps
        self.plan_with_speed = plan_with_speed
        self.num_bev_waypoints = num_bev_waypoints
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.cfg_p_uncond = cfg_p_uncond

        # path / trajectory embeddings + transformer
        self.traj_dim = 4 if not plan_with_speed else 1
        self.transformer = build_head(disentangled_decoder)

        # scene tokenizer
        self.scene_mask_kernel = 1
        self.bev_pooling = bev_pooling
        if bev_pooling is not None:
            assert bev_pooling > 1
            self.bev_pooling = nn.AvgPool2d(kernel_size=bev_pooling, stride=bev_pooling)
            self.scene_mask_kernel *= bev_pooling
        self.bev_unshuffling = bev_unshuffling
        if bev_unshuffling is not None:
            assert bev_unshuffling > 1
            self.bev_unshuffling = nn.PixelUnshuffle(downscale_factor=bev_unshuffling)
            self.scene_mask_kernel *= bev_unshuffling
            # self.unshuffling_projection = nn.Linear(
            #     in_features=d_bev * bev_unshuffling**2, out_features=d_model
            # )
            self.bev_to_token_projection = nn.Linear(
                in_features=d_bev * bev_unshuffling**2, out_features=d_model
            )
        elif d_bev != d_model:
            self.bev_to_token_projection = nn.Linear(
                in_features=d_bev, out_features=d_model
            )
            # self.unshuffling_projection = nn.Linear(
            #     in_features=d_bev, out_features=d_model
            # )
        else:
            self.bev_to_token_projection = nn.Identity()

        self.bev_pe = SinusoidalPositionalEncoding2D(
            len_x=bev_size // self.scene_mask_kernel,
            len_y=bev_size // self.scene_mask_kernel,
            d_model=d_model,
        )
        self.query_pe = nn.Parameter(
            torch.randn(1, num_planning_steps + num_bev_waypoints, d_model)
        )

        # shared embeddings
        self.command_embedding = nn.Embedding(
            num_embeddings=num_commands,
            embedding_dim=d_model,
        )
        self.ego_status_encoder = nn.Sequential(
            SinusoidalPosEmb(d_model // 4),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )

        # path / trajectory decoders
        self.path_decoder = WaypointDecoder(
            input_dim=d_model, hidden_dim=d_model // 2, output_dim=2
        )
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model // 2),
            nn.SiLU(),
            nn.Linear(in_features=d_model // 2, out_features=self.traj_dim),
        )

        if self.plan_with_speed:
            self.rollout = BezierTrajectoryRollout(
                n_bezier=6,
                input_frame_rate=5,
                output_frame_rate=5,
                polyline_len=num_bev_waypoints + 1,
            )

    def forward(
        self,
        # training + inference
        bev_features: torch.Tensor,
        bev_mask: torch.Tensor | None,
        current_speed: torch.Tensor,
        command: torch.Tensor,
        # training (GT)
        bev_waypoints=None,
        bev_waypoints_mask=None,
        planning_traj=None,
        planning_mask=None,
        # critical actor tracking
        critical_actor_centers: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        bs = bev_features.shape[0]
        device = bev_features.device

        x_mask = (
            torch.cat([bev_waypoints_mask, planning_mask], dim=1)
            if bev_waypoints_mask is not None and planning_mask is not None
            else None
        )
        x_padding_mask = torch.logical_not(x_mask) if x_mask is not None else None

        # scene tokens
        scene_tokens, scene_padding_mask = self._tokenize_scene(bev_features, bev_mask)

        # conditioning
        command_embed = self.command_embedding(command)
        ego_status_embed = self.ego_status_encoder(current_speed.unsqueeze(-1))

        # ego status dropout for classifier-free guidance
        if self.cfg_p_uncond > 0.0 and self.training:
            drop_mask = torch.rand(bs, device=device) < self.cfg_p_uncond
            ego_status_embed[drop_mask] = 0.0

        cond = command_embed + ego_status_embed

        # pass through DiT
        attn_weights = []
        out, intermediate_losses = self.transformer(
            q=self.query_pe.repeat(bs, 1, 1),
            kv=scene_tokens,
            bev=bev_features,
            cond=cond,
            q_padding_mask=x_padding_mask,
            kv_padding_mask=scene_padding_mask,
            gt_path=bev_waypoints,
            gt_path_mask=bev_waypoints_mask,
            log_attention_weights=critical_actor_centers is not None,
        )
        xhat = out["q"]
        if (attn_weight := out.get("attn_weights")) is not None:
            attn_weights.append(attn_weight)

        # split path and trajectory embeddings
        path_emb, traj_emb = torch.split(
            xhat, [self.num_bev_waypoints, self.num_planning_steps], dim=1
        )

        # decode
        pred_path = self.path_decoder(path_emb)
        pred_traj = self.trajectory_decoder(traj_emb)

        # [optional] compute losses
        if bev_waypoints is not None and planning_traj is not None:
            losses = self._compute_loss(
                pred_path,
                bev_waypoints,
                pred_traj,
                planning_traj,
                bev_waypoints_mask,
                planning_mask,
            )
            losses.update(intermediate_losses)
        else:
            losses = {}

        if self.plan_with_speed:
            pred_dist = pred_traj.squeeze(-1)

            # compute a roll-out
            pred_traj, _, roll_out_mask = self.rollout(
                pred_path, pred_dist, planning_mask
            )

            # update the planning mask (some timesteps may not be valid)
            if roll_out_mask is not None:
                planning_mask = roll_out_mask

        if len(attn_weights) > 0:
            attn_weights = torch.stack(attn_weights, dim=0)
        else:
            attn_weights = None

        if attn_weights is not None and critical_actor_centers is not None:
            path_attn = attn_weights[:, :, :, : self.num_bev_waypoints]
            traj_attn = attn_weights[:, :, :, self.num_bev_waypoints :]
            critical_attn_path = self._get_critical_actor_attn(
                path_attn, critical_actor_centers
            )
            critical_attn_traj = self._get_critical_actor_attn(
                traj_attn, critical_actor_centers
            )
        else:
            critical_attn_path = critical_attn_traj = None

        results = dict(
            pred_bev_waypoints=pred_path,
            pred_trajectory=pred_traj[..., :2],
            planning_mask=planning_mask,
            attn_weights=attn_weights,
            critical_attn_path=critical_attn_path,
            critical_attn_traj=critical_attn_traj,
        )

        # [optional]: compute metrics
        if bev_waypoints is not None and planning_traj is not None:
            path_l1 = self._compute_waypoint_l1(
                pred_path, bev_waypoints, bev_waypoints_mask
            )
            path_l1_10m = self._compute_waypoint_l1(
                pred_path, bev_waypoints, bev_waypoints_mask, maxlen=10
            )
            path_l1_20m = self._compute_waypoint_l1(
                pred_path, bev_waypoints, bev_waypoints_mask, maxlen=20
            )
            traj_l1 = self._compute_waypoint_l1(
                pred_traj[..., :2], planning_traj[..., :2], planning_mask
            )
            traj_l1_1s = self._compute_waypoint_l1(
                pred_traj[..., :2],
                planning_traj[..., :2],
                planning_mask,
                maxlen=5,
            )
            traj_l1_2s = self._compute_waypoint_l1(
                pred_traj[..., :2],
                planning_traj[..., :2],
                planning_mask,
                maxlen=10,
            )
            losses.update(
                dict(
                    l1_path=path_l1,
                    l1_traj=traj_l1,
                    l1_path_10m=path_l1_10m,
                    l1_path_20m=path_l1_20m,
                    l1_traj_1s=traj_l1_1s,
                    l1_traj_2s=traj_l1_2s,
                )
            )

        return results, losses

    def _tokenize_scene(self, bev: torch.Tensor, bev_mask: torch.Tensor | None):
        bs, hw, c = bev.shape

        # shape as image
        h = w = int(math.sqrt(hw))
        spatial_bev = bev.permute(0, 2, 1).reshape(bs, c, h, w)
        spatial_mask = bev_mask.view(bs, h, w) if bev_mask is not None else None

        # pool
        if self.bev_pooling is not None:
            spatial_bev = self.bev_pooling(spatial_bev)

        # unshuffle
        if self.bev_unshuffling is not None:
            spatial_bev = self.bev_unshuffling(spatial_bev)

        # adapt mask
        if spatial_mask is not None:
            spatial_mask = (
                nn.functional.avg_pool2d(
                    spatial_mask.unsqueeze(1).float(),
                    kernel_size=self.scene_mask_kernel,
                    stride=self.scene_mask_kernel,
                ).squeeze(1)
                > 0.5
            )

        # shape as sequence
        scene_tokens = spatial_bev.flatten(2).permute(0, 2, 1)
        scene_key_padding_mask = (
            spatial_mask.flatten(1) if spatial_mask is not None else None
        )

        # project to model dim
        # scene_tokens = self.unshuffling_projection(scene_tokens)
        scene_tokens = self.bev_to_token_projection(scene_tokens)

        # add PE
        scene_pe = self.bev_pe(scene_tokens).to(bev.dtype)
        scene_tokens = scene_tokens + scene_pe

        return scene_tokens, scene_key_padding_mask

    def _compute_loss(
        self, input_path, target_path, input_traj, target_traj, mask_path, mask_traj
    ):
        input_path = normalize_path(input_path)
        target_path = normalize_path(target_path)
        if self.plan_with_speed:
            input_traj = normalize_trajectory_dist(input_traj)
            target_traj = trajectory_to_distance(target_traj).unsqueeze(-1)
            target_traj = normalize_trajectory_dist(target_traj)
        else:
            input_traj = normalize_trajectory(input_traj)
            target_traj = normalize_trajectory(target_traj)

        loss_path = (
            nn.functional.smooth_l1_loss(
                input_path[mask_path], target_path[mask_path], reduction="none"
            )
            .sum(-1)
            .mean()
        )
        loss_traj = (
            nn.functional.smooth_l1_loss(
                input_traj[mask_traj], target_traj[mask_traj], reduction="none"
            )
            .sum(-1)
            .mean()
        )
        return {
            "loss_path": loss_path * self.loss_weight,
            "loss_traj": loss_traj * self.loss_weight,
        }

    def _compute_waypoint_l1(self, input, target, mask, maxlen: int = None):
        delta = torch.sqrt(torch.sum(torch.square(input - target), dim=-1))
        if maxlen is not None:
            mask = mask.clone()
            mask[..., maxlen:] = False
        l1 = delta[mask].mean()
        return l1

    def _get_critical_actor_attn(
        self, attn_weights: torch.Tensor, critical_actor_centers: torch.Tensor
    ):
        """Compute attention weights for critical actors.

        Args:
            attn_weights (torch.Tensor): Shape: num_denoising_steps x num_layers x B x num_queries x spatial_hw
            critical_actor_centers (torch.Tensor): Shape: B x N x 2
            critical_actor_mask (torch.Tensor): Shape: B x N
        """
        bs = attn_weights.shape[2]
        hw = attn_weights.shape[-1]
        h = w = int(math.sqrt(hw))

        # aggregate attention along query dimension
        attn_weights = attn_weights.mean(
            3
        )  # num_denoising_steps x num_layers x B x spatial_hw

        # transform attention weights into 2D spatial map
        attn_map = attn_weights.permute(2, 0, 1, 3).flatten(1, 2).reshape(bs, -1, h, w)

        # compute sampling points from critical actor centers
        sampling_points = torch.zeros_like(critical_actor_centers)
        sampling_points[..., 0] = (
            -critical_actor_centers[..., 1] / 40
        )  # TODO: remove hardcoded 40
        sampling_points[..., 1] = critical_actor_centers[..., 0] / 40
        sampling_points = sampling_points[:, :, None, :]  # B x N x 1 x 2

        # sample attention weights at critical actor locations
        critical_attn = (
            nn.functional.grid_sample(attn_map, sampling_points)
            .squeeze(-1)
            .permute(0, 2, 1)
        )  # B x N x (num_denoising_steps * num_layers)
        return critical_attn
