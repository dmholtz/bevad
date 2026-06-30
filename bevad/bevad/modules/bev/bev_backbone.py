# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#
#  Additionally modified by dmholtz
# ---------------------------------------------

import os

import torch
import torch.nn as nn
import torchvision
from mmcv.models import DETECTORS, GridMask
from mmcv.models.backbones.base_module import BaseModule
from mmcv.models.bricks.transformer import build_transformer_layer_sequence
from mmcv.models.builder import build_backbone, build_neck

DEBUG_STREAMING = os.environ.get("DEBUG_STREAMING", False) == "1"


@DETECTORS.register_module()
class BevBackbone(BaseModule):
    def __init__(
        self,
        *,
        d_model: int,
        use_grid_mask: bool = False,
        use_dynamics: bool,
        use_cams_embeds: bool,
        img_backbone,
        img_neck,
        bev_encoder,
        bev_size: int,
        bev_range: list[float],
        num_feature_levels: int,
        train_cfg=None,
        **kwargs,
    ):
        super().__init__()

        # BEV config
        self.real_w = bev_range[4] - bev_range[1]
        self.real_h = bev_range[3] - bev_range[0]
        self.bev_h = bev_size
        self.bev_w = bev_size

        self.embed_dims = d_model

        # compatibility attributes
        self.num_cams = 6
        self.use_shift = True

        self.train_cfg = train_cfg

        self.img_backbone = build_backbone(img_backbone)

        if img_neck is not None:
            self.img_neck = build_neck(img_neck)

        if bev_encoder:
            # learnable BEV queries
            self.bev_queries = nn.Parameter(
                torch.randn(1, self.bev_h * self.bev_w, self.embed_dims)
            )

            self.level_embeds = nn.Parameter(
                torch.randn(num_feature_levels, self.embed_dims)
            )
            self.bev_encoder = build_transformer_layer_sequence(bev_encoder)

        self.use_cams_embeds = use_cams_embeds
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(torch.randn(self.num_cams, self.embed_dims))

        self.use_dynamics = use_dynamics
        if self.use_dynamics:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(9, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
            )

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.6, mode=1, prob=0.7
            )

    def extract_img_feat(self, img):
        assert img.dim() == 5
        bs, num_cams, c, h, w = img.shape

        # combine batch and camera dimensions
        img = img.view(bs * num_cams, c, h, w)

        # [optional]: apply grid mask
        if self.use_grid_mask:
            img = self.grid_mask(img)

        # extract image features using backbone
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, torch.Tensor):
            img_feats = [img_feats]

        # extract hierachical features using FPN
        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)

        # reshape image features
        reshaped_feats = []
        for img_feat in img_feats:
            bs, c_, h_, w_ = img_feat.shape
            img_feat = img_feat.view(bs // num_cams, num_cams, c_, h_, w_)
            reshaped_feats.append(img_feat)

        return reshaped_feats

    # @torch.compiler.disable
    def get_bev_features(
        self,
        *,
        mlvl_feats,
        bev_h,
        bev_w,
        grid_length=None,
        prev_bev=None,
        prev_dynamics,
        prev_bev_valid=None,
        memory_frequency=None,
        localization=None,
        dynamics=None,
        img_tf_lidar=None,
        img_shape=None,
        # perfect odometry (for debugging only)
        world_tf_present=None,
        past_tf_world=None,
        **kwargs,
    ):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        device = mlvl_feats[0].device
        bev_queries = self.bev_queries.expand(bs, -1, -1)
        bev_pos = None

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            if (
                past_tf_world is not None
                and world_tf_present is not None
                and DEBUG_STREAMING
            ):
                past_tf_present = past_tf_world @ world_tf_present

                # extract rotation and translation
                shift_meters_iso = past_tf_present[:, :2, 3]  # in meters in ISO

                # convert translation to LiDAR frame and normalize by BEV range
                shift_meters_lidar = torch.zeros_like(shift_meters_iso)
                shift_meters_lidar[:, 0] = -shift_meters_iso[:, 1]
                shift_meters_lidar[:, 1] = shift_meters_iso[:, 0]

                prev_tf_present_2d_priv = (
                    torch.eye(3, device=device).unsqueeze(0).repeat(bs, 1, 1)
                )
                prev_tf_present_2d_priv[:, :2, :2] = past_tf_present[:, :2, :2]
                prev_tf_present_2d_priv[:, :2, 2] = shift_meters_lidar

            if prev_dynamics is not None:
                # compute mean rotation and speed
                mean_yaw_rate = (prev_dynamics[:, 8] + dynamics[:, 8]) / 2.0
                mean_speed = (prev_dynamics[:, 0] + dynamics[:, 0]) / 2.0

                # integrate
                dt = 1 / memory_frequency
                yaw = mean_yaw_rate * dt
                translation_x = mean_speed * dt

                # build transformation matrix from rotation and translation
                prev_tf_present_2d = (
                    torch.eye(3, device=device).unsqueeze(0).repeat(bs, 1, 1)
                )
                prev_tf_present_2d[:, 0, 0] = torch.cos(yaw)
                prev_tf_present_2d[:, 0, 1] = -torch.sin(yaw)
                prev_tf_present_2d[:, 1, 0] = torch.sin(yaw)
                prev_tf_present_2d[:, 1, 1] = torch.cos(yaw)
                prev_tf_present_2d[:, 0, 2] = torch.cos(yaw / 2) * translation_x
                prev_tf_present_2d[:, 1, 2] = torch.sin(yaw / 2) * translation_x

                # convert to LiDAR coordinate system
                shift_iso = prev_tf_present_2d[:, :2, 2]
                shift_lidar = torch.zeros_like(shift_iso)
                shift_lidar[:, 0] = -shift_iso[:, 1]
                shift_lidar[:, 1] = shift_iso[:, 0]
                prev_tf_present_2d[:, :2, 2] = shift_lidar
            else:
                prev_tf_present_2d = (
                    torch.eye(3, device=device).unsqueeze(0).repeat(bs, 1, 1)
                )

        # [optional]: add dynamics embedding
        if self.use_dynamics:
            assert dynamics is not None
            dynamics_emb = self.can_bus_mlp(dynamics).unsqueeze(1)
            bev_queries = bev_queries + dynamics_emb

        # convert list of multi-level features from [BxNxCxHxW] into BxNx[sum(HW)]xC tensor
        spatial_shapes = [(feat.size(3), feat.size(4)) for feat in mlvl_feats]
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device
        )
        mlvl_feats = [feat.permute(0, 1, 3, 4, 2).flatten(2, 3) for feat in mlvl_feats]
        mlvl_feats = [
            feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            for lvl, feat in enumerate(mlvl_feats)
        ]
        stacked_features = torch.cat(mlvl_feats, dim=2)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # [optional] add camera embedding
        if self.use_cams_embeds:
            stacked_features = stacked_features + self.cams_embeds[None, :, None, :].to(
                stacked_features.dtype
            )

        bev_embed = self.bev_encoder(
            bev_queries.to(stacked_features.dtype),
            stacked_features,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            prev_tf_present=prev_tf_present_2d,
            prev_bev_valid=prev_bev_valid,
            img_tf_lidar=img_tf_lidar,
            img_shape=img_shape,
            **kwargs,
        )

        return bev_embed

    def forward(
        self,
        img,
        *,
        img_tf_lidar: torch.Tensor,
        localization: torch.Tensor | None = None,
        dynamics: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        prev_dynamics: torch.Tensor | None = None,
        prev_bev_valid: torch.Tensor | None = None,
        memory_frequency: int | None = None,
        # perfect odometry (for debugging only)
        world_tf_present: torch.Tensor | None = None,
        past_tf_world: torch.Tensor | None = None,
        **kwargs,
    ):
        # extract image features using backbone and FPN
        mlvl_features = self.extract_img_feat(img)

        # compute BEV
        bev = self.get_bev_features(
            mlvl_feats=mlvl_features,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            img_shape=img.shape[-2:],
            img_tf_lidar=img_tf_lidar,
            localization=localization,
            dynamics=dynamics,
            prev_bev=prev_bev,
            prev_dynamics=prev_dynamics,
            prev_bev_valid=prev_bev_valid,
            memory_frequency=memory_frequency,
            # perfect odometry (for debugging only)
            world_tf_present=world_tf_present,
            past_tf_world=past_tf_world,
        )

        return bev
