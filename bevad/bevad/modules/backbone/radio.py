from typing import Literal

import torch
import torch.nn as nn
from mmcv.models import HEADS
from peft import LoraConfig, get_peft_model


@HEADS.register_module()
class RadioBackbone(nn.Module):
    def __init__(self, model_version: str, trainable: bool | Literal["LoRA"]):
        super().__init__()

        self.radio = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=model_version,
            progress=True,
            skip_validation=True,
        )

        if trainable == "LoRA":
            # LoRA configuration
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules="all-linear",
                lora_dropout=0.1,
                bias="none",
            )
            self.radio = get_peft_model(self.radio, lora_config)
            print("Fine-tune RADIO backbone with LoRA")
            self.radio.print_trainable_parameters()
        elif trainable is True:
            print("Unfreeze RADIO backbone parameters")
        elif trainable is False:
            for param in self.radio.parameters():
                param.requires_grad = False
            print("Freezing RADIO backbone parameters")
        else:
            raise ValueError(
                f"Invalid trainable option: {trainable}. Use True, False, or 'LoRA'."
            )

    def forward(self, imgs: torch.Tensor):
        _, spatial_features = self.radio(imgs, feature_fmt="NCHW")
        return spatial_features.contiguous()


@HEADS.register_module()
class RadioNeck(nn.Module):
    """A simple neck that adapts the output of the RADIO backbone to a specified output dimension."""

    def __init__(self, d_input, d_output):
        super().__init__()
        self._conv_adapter = nn.Conv2d(
            d_input, d_output, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: list[torch.Tensor]):
        return [self._conv_adapter(f) for f in x]
