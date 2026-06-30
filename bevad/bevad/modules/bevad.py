from collections import OrderedDict, defaultdict
from typing import Optional

import torch
import torch.distributed as dist
from mmcv.models import DETECTORS
from mmcv.models.backbones.base_module import BaseModule
from mmcv.models.builder import build_head
from mmcv.utils import ConfigDict


@DETECTORS.register_module()
class BEVAd(BaseModule):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        *args,
        # settings
        freeze_bev: bool = False,
        task_loss_weight: dict = defaultdict(lambda: 1.0),
        train_det: bool = True,
        # components
        memory: Optional[ConfigDict] = None,
        bev_backbone: Optional[ConfigDict] = None,
        detection_head: Optional[ConfigDict] = None,
        map_head: Optional[ConfigDict] = None,
        bev_crop: Optional[ConfigDict] = None,
        planning_head: Optional[ConfigDict] = None,
        **kwargs,
    ):
        super().__init__()

        # config
        self.freeze_bev = freeze_bev
        self.task_loss_weight = defaultdict(lambda: 1.0)
        self.task_loss_weight.update(task_loss_weight)
        self.train_det = train_det

        # memory
        self.memory = None
        if memory:
            self.memory = build_head(memory)

        # BEV backbones
        self.bev_backbone = None
        self.proxy_bev_backbone = None
        if bev_backbone:
            self.bev_backbone = build_head(bev_backbone)

        # detection head
        self.detection_head = None
        if detection_head:
            self.detection_head = build_head(detection_head)

        # map head
        self.map_head = None
        if map_head:
            self.map_head = build_head(map_head)

        # BEV augmentation
        self.bev_augmentation = None

        self.bev_crop = None
        if bev_crop:
            self.bev_crop = build_head(bev_crop)

        # planning head
        self.planning_head = None
        if planning_head:
            self.planning_head = build_head(planning_head)

        if self.freeze_bev:
            self.bev_backbone.requires_grad_(False)

    def forward(
        self,
        *,
        # training & inference
        img=None,
        img_tf_lidar=None,
        localization=None,
        dynamics,
        command=None,
        current_speed=None,
        model_frequency: torch.Tensor | int | None = None,
        # training-only (detection)
        gt_labels=None,
        gt_boxes=None,
        gt_masks=None,
        # training-only (mapping)
        map_segmentation=None,
        # training-only (planning)
        planning_traj=None,
        planning_mask=None,
        ego_size=None,
        agent_traj=None,
        agent_yaw=None,
        agent_size=None,
        bev_waypoints=None,
        bev_waypoints_mask=None,
        planning_speed=None,
        speed_two_hot=None,
        all_agents_box=None,
        all_agents_mask=None,
        # misc
        episode_id=None,
        frame_id=None,
        training_progress=None,
        # debugging
        world_tf_ego=None,
        ego_tf_world=None,
        **kwargs,
    ):
        # tensor information
        device = dynamics.device

        outputs = {}
        losses = {}

        # read memory on forward pass start
        memory = {}
        memory_valid = torch.zeros(
            (dynamics.shape[0],), dtype=torch.bool, device=dynamics.device
        )
        if self.memory:
            # determine read frequency
            if isinstance(model_frequency, int):
                read_frequency = model_frequency
            elif isinstance(model_frequency, torch.Tensor):
                read_frequency = model_frequency[0].item()
            else:
                raise ValueError("model_frequency must be int or torch.Tensor")

            memory = self.memory.read_memory(
                read_frequency=read_frequency, expected_batch_size=dynamics.shape[0]
            )

            # check if memory is valid
            last_episode_id = memory.get("episode_id", None)
            last_frame_id = memory.get("frame_id", None)
            expected_frame_delta = read_frequency // self.memory.model_frequency
            if last_episode_id is not None and last_frame_id is not None:
                memory_valid = torch.tensor(
                    [
                        last == current
                        for last, current in zip(last_episode_id, episode_id)
                    ],
                    device=device,
                ) & (last_frame_id == frame_id - expected_frame_delta)

        # BEV backbone
        if self.bev_backbone:
            memory_frequency = self.memory.model_frequency if self.memory else None
            if self.freeze_bev:
                with torch.no_grad():
                    bev = self.bev_backbone(
                        img,
                        img_tf_lidar=img_tf_lidar,
                        localization=localization,
                        dynamics=dynamics,
                        prev_bev=memory.get("bev"),
                        prev_dynamics=memory.get("dynamics"),
                        prev_bev_valid=memory_valid,
                        memory_frequency=memory_frequency,
                        # perfect odometry (for debugging only)
                        world_tf_present=world_tf_ego,
                        past_tf_world=memory.get("ego_tf_world"),
                    )
            else:
                bev = self.bev_backbone(
                    img,
                    img_tf_lidar=img_tf_lidar,
                    localization=localization,
                    dynamics=dynamics,
                    prev_bev=memory.get("bev"),
                    prev_dynamics=memory.get("dynamics"),
                    prev_bev_valid=memory_valid,
                    memory_frequency=memory_frequency,
                    # perfect odometry (for debugging only)
                    world_tf_present=world_tf_ego,
                    past_tf_world=memory.get("ego_tf_world"),
                )
            bev = bev.to(dtype=img.dtype)
        else:
            raise NotImplementedError("No BEV backbone is provided.")

        outputs["bev"] = bev

        # detection head
        object_features = None
        object_mask = None
        if self.detection_head and self.train_det:
            detection_results = self.detection_head(
                bev,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                gt_masks=gt_masks,
                all_agents_box=all_agents_box,
                all_agents_mask=all_agents_mask,
            )
            if "loss" in detection_results:
                detection_loss = self.loss_weighted_and_prefixed(
                    detection_results["loss"], prefix="det"
                )
                losses.update(detection_loss)

            for k, v in detection_results.items():
                if k.startswith("det."):
                    outputs[k] = v

            if "queries" in detection_results:
                object_features = detection_results["queries"]
            if "object_mask" in detection_results:
                object_mask = detection_results["object_mask"]

            outputs["detection"] = {
                k: v for k, v in detection_results.items() if k.startswith("all_")
            }

        # mapping
        if self.map_head:
            map_output = self.map_head(bev, map_segmentation=map_segmentation)

            if "loss" in map_output:
                map_loss = self.loss_weighted_and_prefixed(
                    map_output["loss"], prefix="map"
                )
                losses.update(map_loss)

            outputs["map_segmentation"] = map_output["map_segmentation"]

        if self.bev_crop:
            pred_map_segmentation = outputs.get("map_segmentation")
            bev_mask = self.bev_crop(bev, map_segmentation=pred_map_segmentation)
        else:
            bev_mask = None

        # planning
        if self.planning_head is not None:
            planning_outputs, planning_losses = self.planning_head(
                bev_features=bev,
                bev_mask=bev_mask,
                current_speed=current_speed,
                command=command,
                object_features=object_features,
                object_mask=object_mask,
                # supervision
                bev_waypoints=bev_waypoints,
                bev_waypoints_mask=bev_waypoints_mask,
                planning_traj=planning_traj,
                planning_speed=planning_speed,
                speed_two_hot=speed_two_hot,
                planning_mask=planning_mask,
                ego_size=ego_size,
                agent_traj=agent_traj,
                agent_yaw=agent_yaw,
                agent_size=agent_size,
                training_progress=training_progress,
            )
            outputs["planning"] = planning_outputs
            planning_losses = self.loss_weighted_and_prefixed(
                planning_losses, prefix="planning"
            )
            losses.update(planning_losses)

        # update memory on forward pass end
        if self.memory:
            write_frequency = read_frequency
            memory_data = {
                "bev": bev.detach().clone(),
                "dynamics": dynamics.detach().clone(),
                "episode_id": episode_id,
                "frame_id": frame_id,
                # perfect odometry (for debugging only)
                "ego_tf_world": ego_tf_world,
            }
            self.memory.write_memory(data=memory_data, write_frequency=write_frequency)

        return outputs, losses

    def loss_weighted_and_prefixed(self, loss_dict, prefix=""):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
        return loss_dict

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

        return loss, log_vars
