from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
from mmcv.models import HEADS
from torchvision.ops.focal_loss import sigmoid_focal_loss


@HEADS.register_module()
class SemanticSegmentationMapDecoder(nn.Module):
    """A fully-convolutional BEV-to-SemanticMap decoder head."""

    def __init__(
        self,
        bev_resolution: int,
        d_bev: int,
        map_resolution: int,
        d_map: int,
        layer_weights: dict,
    ):
        super().__init__()
        self.d_map = d_map
        self.bev_resolution = bev_resolution

        assert map_resolution % bev_resolution == 0, (
            f"BEV resolution ({bev_resolution}) does not divide map resolution ({map_resolution})."
        )
        upsampling_factor = map_resolution // bev_resolution

        self.segmenation_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=d_bev,
                out_channels=d_bev // 8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=d_bev // 8, out_channels=d_map, kernel_size=1, stride=1
            ),
            nn.Upsample(
                scale_factor=upsampling_factor, mode="bilinear", align_corners=False
            ),
        )

        self.layer_weights = defaultdict(lambda: 1.0, layer_weights)

    def forward(self, bev, map_segmentation=None):
        bev_spatial = bev.view(
            bev.shape[0], self.bev_resolution, self.bev_resolution, -1
        )
        bev_spatial = bev_spatial.permute(
            0, 3, 1, 2
        ).contiguous()  # Change to (B, C, H, W)

        # predict logits per map layer
        map_logits = self.segmenation_decoder(bev_spatial)

        output = {
            "map_segmentation": self._decode_map_logits(map_logits),
        }

        if map_segmentation is not None:
            # training
            output["loss"] = self._compute_loss(map_logits, map_segmentation)

        # inference
        return output

    def _decode_map_logits(self, map_logits):
        """Decode map logits into probabilites per map layer."""
        return F.sigmoid(map_logits)

    def _compute_loss(self, map_logits, map_segmentation):
        """Compute the map segmentation loss as BCE for each layer."""
        layer_loss = sigmoid_focal_loss(
            map_logits, map_segmentation, reduction="none"
        ).mean(dim=(0, 2, 3))
        return {
            f"loss_bce_{layer}": loss * self.layer_weights[layer]
            for loss, layer in zip(layer_loss, ("driveable", "lane", "stop"))
        }
