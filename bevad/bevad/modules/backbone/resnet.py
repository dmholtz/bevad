import torch
import torch.nn as nn
import torchvision
from mmcv.models import HEADS
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights


def freeze_layers(module, frozen_layers: list[str]):
    for name, parameter in module.named_parameters():
        if any(name.startswith(layer) for layer in frozen_layers):
            parameter.requires_grad = False
            print(f"Freezing {name}")


@HEADS.register_module()
class ResNetBackbone(nn.Module):
    def __init__(
        self, model_version: str, return_layers: list[str], frozen_layers: list[str]
    ):
        super().__init__()

        if model_version == "50":
            resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_version == "101":
            resnet = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet depth")
        freeze_layers(resnet, frozen_layers)
        self.img_backbone = IntermediateLayerGetter(
            resnet, return_layers={i: i for i in return_layers}
        )

    def forward(self, imgs: torch.Tensor):
        spatial_features = list(self.img_backbone(imgs).values())
        return spatial_features
