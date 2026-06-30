import argparse
import os

import lightning as L
import torch
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from mmcv.datasets import build_dataset
from mmcv.utils import Config
from torch.utils.data import DataLoader

from bevad.data.sampler import DistributedStreamSampler
from bevad.modules.bevad_lightning import BevAdLightning

# global accelerator settings (only relevant for A100)
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.enable_flash_sdp(True)


def train_val(config_file: str, resume_ckpt: str | None = None):
    cfg = Config.fromfile(config_file)

    # logger
    logger = WandbLogger(
        entity="ad-research",
        project="bevad",
        job_type="bevad-train",
        tags=cfg.logger.tags,
        log_model="all",  # save checkpoint to wandb
        save_dir=".",
    )
    logger.experiment  # to initialize the logger

    # model
    if resume_ckpt is not None:
        bevad = BevAdLightning.load_from_checkpoint(
            resume_ckpt, config=cfg, strict=False
        )
    else:
        # build model from scratch
        bevad = BevAdLightning(config=cfg)
        bevad.model.init_weights()

    # dataset
    training_dataset = build_dataset(cfg.data.train)
    validation_dataset = build_dataset(cfg.data.val)

    # sampler
    batch_size_train = cfg.data.batch_size_train
    batch_size_val = cfg.data.batch_size_val
    if cfg.data.streaming:
        # sampler for streaming training & validation
        sampler_train = DistributedStreamSampler(
            training_dataset, batch_size_train, shuffle=True
        )
        sampler_val = DistributedStreamSampler(
            validation_dataset, batch_size_val, shuffle=False
        )

        # dataloader settings for compatibility
        batch_size_train = 1  # batch size is handled by the batch sampler
        batch_size_val = 1  # batch size is handled by the batch sampler
        use_distributed_sampler = (
            False  # DistributedStreamSampler handles both DDP and non-DDP
        )
    else:
        # sampler for non-temporal training & validation
        sampler_train = None
        sampler_val = None
        use_distributed_sampler = True

    # data loader
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size_train,
        batch_sampler=sampler_train,
        num_workers=cfg.data.workers_per_gpu,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size_val,
        batch_sampler=sampler_val,
        num_workers=cfg.data.workers_per_gpu,
        pin_memory=True,
    )

    # progress bar
    class CustomProgressBar(TQDMProgressBar):
        def get_metrics(self, *args, **kwargs):
            # don't show the version number
            items = super().get_metrics(*args, **kwargs)
            items.pop("v_num", None)
            return items

    progress_bar = CustomProgressBar(leave=True)

    # precision: use bf16 if available, otherwise use 16-bit mixed precision
    precision = "bf16-mixed" if os.environ.get("AVOID_BF16") is None else "16-mixed"

    # training & validation
    trainer = L.Trainer(
        # strategy='ddp_find_unused_parameters_true',
        max_epochs=cfg.trainer.max_epochs,
        precision=precision,
        callbacks=[progress_bar],
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=35,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        use_distributed_sampler=use_distributed_sampler,
    )
    trainer.fit(
        model=bevad,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train BEVAD model")
    arg_parser.add_argument("config", type=str, help="Path to the config file")
    arg_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to the checkpoint to resume training from",
    )

    args = arg_parser.parse_args()
    config_file = args.config
    resume_ckpt = args.resume

    train_val(config_file, resume_ckpt=resume_ckpt)
