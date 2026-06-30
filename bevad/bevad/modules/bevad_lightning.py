from collections import OrderedDict, defaultdict

import lightning as L

import schedulefree
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import grad_norm

from mmcv.models import build_model



class BevAdLightning(L.LightningModule):
    """BevAdLightning is a LightningModule that wraps around a BEVAD model."""

    def __init__(self, config):
        super().__init__()

        self.model = build_model(config.model)
        self.cfg = config


    def training_step(self, *args, **kwargs):
        self.optimizers().train()

        progress = self.global_step / self.trainer.estimated_stepping_batches
        _, losses = self.model(**args[0], training_progress=progress)
        loss, log_vars = self._parse_losses(losses)

        for name, value in log_vars.items():
            prog_bar = name in (
                "loss",
                "det.loss_cls",
                "det.loss_bbox",
                "planning.loss_traj",
            )
            self.log(f"train/{name}", value, prog_bar=prog_bar)
        self.log("data_time", args[0]["data_time"].mean(), prog_bar=True)

        return loss

    def validation_step(self, *args, **kwargs):
        opt = self.optimizers()
        if hasattr(
            opt, "eval"
        ):
            opt.eval()

        outputs, losses = self.model(**args[0], training_progress=1)
        if len(losses) > 0:
            _, log_vars = self._parse_losses(losses)
        else:
            log_vars = {}

        for name, value in log_vars.items():
            prog_bar = name in (
                "loss",
                "det.loss_cls",
                "det.loss_bbox",
                "planning.l1_traj",
            )
            self.log(
                f"val/{name}",
                value,
                prog_bar=prog_bar,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(args[0]["dynamics"]),
            )
        self.log(
            "data_time",
            args[0]["data_time"].mean(),
            prog_bar=True,
            batch_size=len(args[0]["dynamics"]),
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # default learning rate
        lr = self.cfg.lightning.lr

        # layer-specific learning rates
        layer_lr = {
            "img_backbone": lr * 0.1,
        }

        # group parameters by layer
        grouped_params = defaultdict(list)
        for param_name, param in self.model.named_parameters():
            is_default_lr = True
            for layer in layer_lr.keys():
                if layer in param_name:
                    grouped_params[layer].append(param)
                    is_default_lr = False
                    break
            if is_default_lr:
                grouped_params["default"].append(param)

        # create optimizer configuration
        opt_config = []
        for group, params in grouped_params.items():
            if group == "default":
                opt_config.append({"params": params})
            else:
                opt_config.append({"params": params, "lr": layer_lr[group]})

        # build the optimizer
        return schedulefree.AdamWScheduleFree(opt_config, lr=lr, weight_decay=0.01)

    def on_train_epoch_start(self):
        self.optimizers().train()

    def on_validation_epoch_start(self):
        self.optimizers().eval()

    def on_before_optimizer_step(self, optimizer):
        total_grad_norm = grad_norm(self.model, norm_type=2)["grad_2.0_norm_total"]
        self.log("train/grad_norm", total_grad_norm)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_varsW
