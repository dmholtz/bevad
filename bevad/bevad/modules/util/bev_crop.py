import math

import torch
import torch.nn as nn
from mmcv.models import HEADS
from torchvision.transforms.v2.functional import gaussian_blur


@HEADS.register_module()
class BevCrop(nn.Module):
    def __init__(
        self, input_size: int, cells_ahead: int, cells_leftright: int, cells_behind: int
    ):
        super().__init__()

        self.input_size = input_size
        self.half_size = input_size // 2
        self.cells_ahead = cells_ahead
        self.cell_leftright = cells_leftright
        self.cells_behind = cells_behind

    def forward(self, bev, **kwargs):
        bs, hw, _ = bev.shape
        h = int(math.sqrt(hw))
        bev_mask = torch.zeros_like(bev[..., 0], dtype=torch.bool)

        bev_mask = bev_mask.view(bs, h, h)

        # pad behind
        pad_behind = self.half_size - self.cells_behind
        bev_mask[:, :pad_behind, :] = True

        # pad ahead
        pad_ahead = self.half_size + self.cells_ahead
        bev_mask[:, pad_ahead:, :] = True

        # pad left
        pad_left = self.half_size - self.cell_leftright
        bev_mask[:, :, :pad_left] = True

        # pad right
        pad_right = self.half_size + self.cell_leftright
        bev_mask[:, :, pad_right:] = True

        return bev_mask.view(bs, hw)


@HEADS.register_module()
class MapAwareBevCrop(nn.Module):
    def __init__(self, threshold: float, blur_kernel_size: int):
        super().__init__()

        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size

    def forward(self, bev, map_segmentation):
        _, hw_bev, _ = bev.shape
        h_bev = int(math.sqrt(hw_bev))

        # rescale the map segmentation to the bev resolution
        map_segmentation = nn.functional.interpolate(
            map_segmentation.detach(),
            size=(h_bev, h_bev),
            mode="bilinear",
            align_corners=False,
        )

        # combine all channels and threshold
        map_mask = torch.max(map_segmentation, dim=1)[0] < self.threshold

        # extend map mask a bit with gaussian blur
        extended_map_mask = (
            gaussian_blur(
                torch.logical_not(map_mask).unsqueeze(1).float(),
                kernel_size=self.blur_kernel_size,
            )
            < self.threshold
        )

        return extended_map_mask.flatten(1)
