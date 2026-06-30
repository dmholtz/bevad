import os
import pickle
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from os import path as osp
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torchvision.transforms.v2 as transforms
from bevad_sim.data_interface import CoreContainer
from bevad_sim.data_interface.episode_io import EpisodeIo
from bevad_sim.data_interface.tce import TrafficControlElements
from bevad_sim.data_interface.world_state import WorldState
from filelock import FileLock

# import cv2
# from pyquaternion import Quaternion
from mmcv.datasets import DATASETS
from PIL import Image
from torch.utils.data import Dataset

from bevad.data.map_utils import MapExtractor

ego_tf_lidar = np.array(
    [
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


@DATASETS.register_module()
class CoreDataset(Dataset):
    def __init__(
        self,
        # paths
        episode_base: str,
        index_file: str,
        map_file: str,
        # common
        oversample: bool,
        camera_augmentation: bool,
        max_stream_length: int,
        model_frequency: int,
        # config flags
        with_perception: bool,
        with_bev_image: bool,
        with_objects: bool,
        with_map: bool,
        with_planning: bool,
        # BEV config:
        bev_range: list[float],
        # perception config
        backbone_type: str,
        img_resolution: tuple[int, int],
        # object detection config
        class_names: list[str],
        class_mapping: dict[str, str],
        # map config
        map_resolution: int,
        # planning config
        planning_frame_rate: int,
        planning_frames: int,
        num_bev_waypoints: int,
        bev_waypoint_distance: float,
        require_replanning: bool,
        fix_lane_change_commands: bool,
    ):
        # common config
        self.dataset_frame_rate = 10
        self.oversample = oversample
        self.camera_augmentation = camera_augmentation
        self.streaming = max_stream_length is not None and max_stream_length > 0
        self.max_stream_length = max_stream_length
        self.model_frequency = model_frequency
        self.dtype = torch.float16

        self.episode_base = episode_base
        self._load_index(index_file)
        self.episode_cache = "/tmp/episode_cache"
        self.episode_cache_lockfile = os.path.join(self.episode_cache, ".lock")
        Path(self.episode_cache).mkdir(parents=True, exist_ok=True)

        # config flags
        self.with_perception = with_perception
        self.with_bev_image = with_bev_image
        self.with_odometry = True
        self.with_objects = with_objects
        self.with_map = with_map
        self.with_planning = with_planning

        # perception config
        self.camera_names = [
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ]
        if backbone_type not in ("ResNet", "DINO", "RADIO"):
            raise ValueError(
                f"Unsupported backbone type: {backbone_type}. Supported types are 'ResNet', 'DINO', and 'RADIO'."
            )
        self.backbone_type = backbone_type
        if img_resolution not in ((448, 800), (896, 1600)):
            raise ValueError(
                f"Unsupported image resolution: {img_resolution}. Supported resolutions are (448, 800) and (896, 1600)."
            )
        self.img_resolution = img_resolution
        self.img_cache = "/tmp/image_cache"
        Path(self.img_cache).mkdir(parents=True, exist_ok=True)

        # objects config
        self.max_num_objects = 100
        self.bev_range = bev_range
        self.class_names = class_names
        self.class2id = {name: i for i, name in enumerate(class_names)}
        self.class_mapping = class_mapping

        # map config
        self.map_resolution = map_resolution
        if self.with_map:
            with open(map_file, "rb") as f:
                map_infos = pickle.load(f)
            self.map_extractor = MapExtractor(map_infos)

        # planning config
        self.planning_frame_rate = planning_frame_rate
        self.planning_frames = planning_frames
        self.num_bev_waypoints = num_bev_waypoints
        self.bev_waypoint_distance = bev_waypoint_distance
        self.require_replanning = require_replanning
        self.fix_lane_change_commands = fix_lane_change_commands

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, index):
        # check the index type
        if isinstance(index, int):
            # this is the normal setting
            index, epoch = index, None
        elif isinstance(index, tuple):
            # for streaming training, the sampler also provides the current epoch for ensuring deterministic
            # randomness within a stream
            index, epoch = index

        t0 = time.time()

        data = None
        while data is None:
            data = self._load_data(index, epoch)

            # pick another random index if the data is None
            index = random.randint(0, len(self) - 1) % len(self)

        data_time = time.time() - t0
        data["data_time"] = np.array([data_time], dtype=np.float32)

        data["model_frequency"] = self.dataset_frame_rate

        return data

    def _load_index(self, index_file: str):
        df = pl.read_csv(index_file)

        if self.oversample:
            # 1) find and count interesting samples
            num_command, num_acc, num_yaw_rate = 0, 0, 0
            for row in df.iter_rows():
                _, _, command, _, acc, yaw_rate = row
                if command in (5, 6):
                    num_command += 1
                elif acc > 2.5 or acc < -10:
                    num_acc += 1
                elif yaw_rate > 0.05 or yaw_rate < -0.05:
                    num_yaw_rate += 1

            # 2) compute histograms and calculate oversampling factors
            acc_bins = np.linspace(-15, 15, 10)
            acc_count, _ = np.histogram(df["acc"], bins=acc_bins)
            yaw_bins = np.linspace(-0.5, 0.5, 10)
            yaw_count, _ = np.histogram(df["yaw_rate"], bins=yaw_bins)
            command_bins = np.linspace(1, 7, 7)
            command_count, _ = np.histogram(df["command"], bins=command_bins)

            def compute_weights(count, total):
                count = count.clip(1, None)  # avoid division by zero
                return np.floor(np.log10(total / count))

            acc_weights = compute_weights(acc_count, len(df))
            yaw_weights = compute_weights(yaw_count, len(df))
            command_weights = compute_weights(command_count, len(df))

            # 3) append oversampled samples to the index
            additional_rows = []
            for row in df.iter_rows():
                scene, frame, command, speed, acc, yaw_rate = row
                command_idx = (np.argmin(command >= command_bins) - 1).clip(
                    0, len(command_weights) - 1
                )
                command_weight = command_weights[command_idx]
                acc_idx = (np.argmin(acc > acc_bins) - 1).clip(0, len(acc_weights) - 1)
                acc_weight = acc_weights[acc_idx]
                yaw_idx = (np.argmin(yaw_rate > yaw_bins) - 1).clip(
                    0, len(yaw_weights) - 1
                )
                yaw_weight = yaw_weights[yaw_idx]
                total_weight = max(command_weight, max(acc_weight, yaw_weight))
                if total_weight > 0:
                    additional_rows.extend(
                        [(scene, frame, command, speed, acc, yaw_rate)]
                        * int(total_weight)
                    )

            if additional_rows:
                additional_df = pl.DataFrame(
                    additional_rows,
                    schema=[
                        "episode_id",
                        "frame_id",
                        "command",
                        "speed",
                        "acc",
                        "yaw_rate",
                    ],
                )
                df = df.vstack(additional_df)

        self.sample_index = df

        # reference implementation for building streams from samples
        if self.streaming:
            self.streams = []
            self.id_to_stream = {}  # maps sample IDs to stream IDs
            self.id_to_stream_pos = {}  # maps sample IDs to index within a stream
            current_stream = []
            for i, row in enumerate(df.iter_rows()):
                scene, frame, command, speed, acc, yaw_rate = row

                # check if new stream needs to be created
                if len(current_stream) > (self.max_stream_length + i % 4):
                    self.streams.append(current_stream)
                    current_stream = []
                elif (
                    len(current_stream) > 0
                    and df["episode_id"][current_stream[0]] != scene
                ):
                    self.streams.append(current_stream)
                    current_stream = []

                current_stream.append(i)
                self.id_to_stream[i] = len(self.streams)
                self.id_to_stream_pos[i] = len(current_stream) - 1

            if len(current_stream) > 0:
                self.streams.append(current_stream)

    def _load_episode(self, sample_meta: dict[str, Any]) -> CoreContainer:
        episode_name = sample_meta["episode_id"][0]
        mounted_episode_path = os.path.join(self.episode_base, episode_name)
        cached_episode_path = os.path.join(self.episode_cache, episode_name)
        cached_episode_file = os.path.join(cached_episode_path, "ep_data.zstd")

        # check if the episode is already cached
        with FileLock(self.episode_cache_lockfile):
            cache_exists = os.path.exists(cached_episode_file)

        if cache_exists:
            # read file w/o lock
            episode = EpisodeIo.read_episode(cached_episode_path, load_payload=False)
        else:
            # read file from mounted path w/o lock
            episode = EpisodeIo.read_episode(mounted_episode_path, load_payload=False)

            # save the episode to the cache unless another process did it already
            with FileLock(self.episode_cache_lockfile):
                if not os.path.exists(cached_episode_file):
                    Path(cached_episode_path).mkdir(exist_ok=False, parents=True)
                    shutil.copyfile(
                        os.path.join(mounted_episode_path, "ep_data.zstd"),
                        cached_episode_file,
                    )

        return episode

    def _load_data(
        self, sample_index: int, epoch: int | None = None
    ) -> dict[str, Any] | None:
        sample_meta = self.sample_index[sample_index]
        episode = self._load_episode(sample_meta)
        frame_idx = sample_meta["frame_id"][0]

        # pre-loading check
        if not self._sample_is_ok(episode, frame_idx):
            return None

        data = {}

        # miscellaneous data
        world_tf_ego = episode.world_state.transform[0, frame_idx, 0]
        ego_tf_world = np.linalg.inv(world_tf_ego)
        sample_token = f"{episode.episode_meta.episode_id[0]}-{frame_idx}"
        data["sample"] = sample_token
        data["episode_id"] = episode.episode_meta.episode_id[0]
        data["frame_id"] = frame_idx

        if self.camera_augmentation:
            # check if camera augmentation needs to be computed in a streaming-safe manner
            if self.streaming:
                # in streaming mode, we need same augmentation for all samples in the stream
                # otherwise temporal modelling will be compromised
                stream_id = self.id_to_stream[sample_index]
                group_frequency = self.dataset_frame_rate // self.model_frequency
                stream_group = self.id_to_stream_pos[sample_index] % group_frequency
                seed = int(epoch * 2**28 + stream_id * 2**8 + stream_group) % (2**32)
            else:
                seed = int(time.time() * 1e6) % (2**32)

            # build a random number generator for this sample's augmentation
            rng = np.random.default_rng(seed)

            yaw = rng.uniform(-np.pi / 8, np.pi / 8)
            y_offset = rng.uniform(-0.75, 0.75)
            ego_tf_aug = np.eye(4, dtype=np.float32)
            ego_tf_aug[:3, :3] = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            ego_tf_aug[:3, 3] = np.array([0, y_offset, 0], dtype=np.float32)
            aug_tf_ego = np.linalg.inv(ego_tf_aug)

            # apply augmentation to the ego-world transformations
            world_tf_ego = world_tf_ego @ ego_tf_aug
            ego_tf_world = aug_tf_ego @ ego_tf_world
        else:
            ego_tf_aug = np.eye(4, dtype=np.float32)

        if self.with_odometry:
            odometry = self._load_odometry(episode, frame_idx)
            data.update(odometry)

            # perfect odometry (for debugging only)
            perfect_odometry = {
                "world_tf_ego": world_tf_ego.astype(np.float32),
                "ego_tf_world": ego_tf_world.astype(np.float32),
            }
            data.update(perfect_odometry)

        if self.with_objects:
            objects = self._load_objects(episode, frame_idx, ego_tf_world)
            if objects is None:
                return None
            data.update(objects)

        if self.with_perception:
            perception = self._load_perception(
                episode, frame_idx, ego_tf_aug=ego_tf_aug
            )
            data.update(perception)

        if self.with_bev_image:
            data["bev_image"] = self._load_bev_image(episode, frame_idx)

        if self.with_map:
            map_data = self._load_map(episode, world_tf_ego)
            data.update(map_data)

        if self.with_planning:
            planning = self._load_planning(
                episode, frame_idx, world_tf_ego, ego_tf_world
            )
            data.update(planning)

        # post-loading check
        if "bev_waypoints_mask" in data and not np.any(data["bev_waypoints_mask"]):
            return None
        if "planning_mask" in data and not np.any(data["planning_mask"]):
            return None

        return data

    def _sample_is_ok(self, episode: CoreContainer, frame_idx: int) -> bool:
        if self.with_objects:
            # there is at least one object in the scene
            vis_objects = (
                episode.world_state.is_valid[0, frame_idx, 1:]
                & episode.world_state.is_visible[0, frame_idx, 1:]
            )
            vis_tce = (
                episode.tce.is_valid[0, frame_idx]
                & episode.tce.is_visible[0, frame_idx]
            )
            if np.sum(vis_objects) + np.sum(vis_tce) == 0:
                return False

        if self.with_planning:
            # there is at least one future planning step
            future_idx = self.dataset_frame_rate // self.planning_frame_rate
            if frame_idx + future_idx >= episode.world_state.shape[1]:
                return False

            # there is at least one future BEV waypoint
            current_pos = episode.world_state.transform[0, frame_idx, 0][:3, 3]
            final_pos = episode.world_state.transform[0, -1, 0][:3, 3]
            if (
                np.linalg.norm(current_pos - final_pos)
                < self.bev_waypoint_distance * 1.1
            ):
                return False

        return True

    def _load_perception(
        self, episode: CoreContainer, frame_idx: int, ego_tf_aug: np.ndarray
    ) -> dict[str, Any]:
        cam_obs = episode.camera_observations["rgb"][0:1, frame_idx : frame_idx + 1]

        cam2tf = {}
        cam2path = {}
        for (t_idx, c_idx), filename in cam_obs.get_idx_filename_pairs(
            base_path=os.path.join(
                self.episode_base, episode.episode_meta.episode_id[0]
            )
        ).items():
            cam_name = cam_obs.sensor_names[c_idx]

            cam_tf_ego = cam_obs.extrinsics[0, t_idx, c_idx]
            intrinsic = cam_obs.intrinsics[0, t_idx, c_idx]
            if self.img_resolution == (448, 800):
                # TODO: fix me this is a hack to compensate for the image resize
                intrinsic[0, 2] /= 2
                intrinsic[1, 2] /= 2
                intrinsic[0, 0] /= 2
                intrinsic[1, 1] /= 2
            elif self.img_resolution == (896, 1600):
                # no intrinisic adaption needed
                pass
            else:
                raise ValueError(
                    f"Unsupported image resolution: {self.img_resolution}. Supported resolutions are (448, 800) and (896, 1600)."
                )
            img_tf_cam = np.eye(4, dtype=np.float32)
            img_tf_cam[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            img_tf_lidar = img_tf_cam @ cam_tf_ego @ ego_tf_aug @ ego_tf_lidar

            cam2tf[cam_name] = img_tf_lidar
            cam2path[cam_name] = os.path.join(
                episode.episode_meta.episode_id[0], filename
            )

        img_tf_lidar = np.stack(list(cam2tf[c] for c in self.camera_names), axis=0)
        img_paths = [cam2path[c] for c in self.camera_names]

        images = self._get_images(img_paths)

        return {
            "img_tf_lidar": img_tf_lidar,
            "img": images,
        }

    def _load_bev_image(self, episode: CoreContainer, frame_idx: int) -> str:
        bev = episode.camera_observations["bev"]
        episode_id = episode.episode_meta.episode_id[0]
        for (t_idx, c_idx), filename in bev.get_idx_filename_pairs(None).items():
            if t_idx == frame_idx:
                return os.path.join(self.episode_base, episode_id, filename)

    def _load_odometry(self, episode: CoreContainer, frame_idx: int) -> dict[str, Any]:
        odometry = episode.odometry
        world_state = episode.world_state

        localization = np.zeros(8, dtype=np.float32)
        # TODO: add localization data

        dynamics = np.zeros(9, dtype=np.float32)
        dynamics[0] = world_state.dynamics[0, frame_idx, 0, 0]
        # TODO: handle NaN values in dynamics
        dynamics[3:6] = odometry.acceleration[0, frame_idx]
        dynamics[6:9] = odometry.angular_velocity[0, frame_idx]

        speed = world_state.dynamics[0, frame_idx, 0, 0]
        noisy_speed = np.clip(speed + np.random.rand() * 0.4 - 0.2, 0, 20).astype(
            np.float32
        )

        return {
            "localization": localization,
            "dynamics": dynamics,
            "current_speed": noisy_speed,
        }

    def _load_objects(
        self, episode: CoreContainer, frame_idx: int, ego_tf_world: np.ndarray
    ) -> dict[str, Any]:
        objects = np.zeros((self.max_num_objects, 9), dtype=np.float32)
        labels = np.zeros((self.max_num_objects,), dtype=np.int64)
        mask = np.zeros((self.max_num_objects,), dtype=bool)

        ws = episode.world_state
        tce = episode.tce

        # traffic control elements
        world_tf_tce = tce.transform[0, frame_idx]
        tce_dynamics = np.zeros((tce.shape[2], 4))
        tce_is_valid = tce.is_valid[0, frame_idx].astype(bool)
        tce_is_visible = tce.is_visible[0, frame_idx].astype(bool)
        tce_sizes = tce.extent[0, frame_idx]
        ego_tf_tce = ego_tf_world[np.newaxis, :, :] @ world_tf_tce
        tce_pos = ego_tf_tce[:, :3, 3]
        tce_yaw = np.arctan2(ego_tf_tce[:, 1, 0], ego_tf_tce[:, 0, 0])
        tce_labels, valid_tce_labels = self._extract_label(tce, frame_idx)
        in_range = (
            (tce_pos[:, 0] >= self.bev_range[0])
            & (tce_pos[:, 1] >= self.bev_range[1])
            & (tce_pos[:, 0] <= self.bev_range[3])
            & (tce_pos[:, 1] <= self.bev_range[4])
        )
        tce_is_valid = tce_is_valid & tce_is_visible & in_range & valid_tce_labels

        # actors
        world_tf_actors = ws.transform[0, frame_idx, 1:]
        actor_dynamics = ws.dynamics[0, frame_idx, 1:]
        actor_is_valid = ws.is_valid[0, frame_idx, 1:].astype(bool)
        actor_is_visible = ws.is_visible[0, frame_idx, 1:].astype(bool)
        actor_sizes = ws.extent[0, frame_idx, 1:]
        ego_tf_actors = ego_tf_world[np.newaxis, :, :] @ world_tf_actors
        actor_pos = ego_tf_actors[:, :3, 3]
        actor_yaw = np.arctan2(ego_tf_actors[:, 1, 0], ego_tf_actors[:, 0, 0])
        actor_labels, valid_actor_labels = self._extract_label(ws, frame_idx)
        actor_labels = actor_labels[1:]  # remove ego label
        valid_actor_labels = valid_actor_labels[1:]  # remove ego label
        in_range = (
            (actor_pos[:, 0] >= self.bev_range[0])
            & (actor_pos[:, 1] >= self.bev_range[1])
            & (actor_pos[:, 0] <= self.bev_range[3])
            & (actor_pos[:, 1] <= self.bev_range[4])
        )
        actor_is_valid = (
            actor_is_valid & actor_is_visible & in_range & valid_actor_labels
        )

        if np.sum(tce_is_valid) + np.sum(actor_is_valid) == 0:
            return None

        # concatenate TCE and actors to objects
        num_objects = min(
            self.max_num_objects, np.sum(tce_is_valid) + np.sum(actor_is_valid)
        )
        object_is_valid = np.concatenate((tce_is_valid, actor_is_valid), axis=0)
        object_pos = np.concatenate((tce_pos, actor_pos), axis=0)
        object_dynamics = np.concatenate((tce_dynamics, actor_dynamics), axis=0)
        object_sizes = np.concatenate((tce_sizes, actor_sizes), axis=0)
        object_yaw = np.concatenate((tce_yaw, actor_yaw), axis=0)
        object_labels = np.concatenate((tce_labels, actor_labels), axis=0)

        # filter by valid objects
        object_pos = object_pos[object_is_valid][:num_objects]
        object_dynamics = object_dynamics[object_is_valid][:num_objects]
        object_sizes = object_sizes[object_is_valid][:num_objects]
        object_yaw = object_yaw[object_is_valid][:num_objects]
        object_labels = object_labels[object_is_valid][:num_objects]

        # build box (in LiDAR frame)
        objects[:num_objects, 0] = -object_pos[:num_objects, 1]
        objects[:num_objects, 1] = object_pos[:num_objects, 0]
        objects[:num_objects, 2] = object_pos[:num_objects, 2]
        objects[:num_objects, 3:6] = object_sizes[:num_objects]
        objects[:num_objects, 6] = object_yaw[:num_objects]
        objects[:num_objects, 7] = object_dynamics[:num_objects, 0]
        # we do not set vy

        # build labels
        labels[:num_objects] = object_labels[:num_objects]

        # build mask
        mask[:num_objects] = True

        return {
            "gt_boxes": objects,
            "gt_labels": labels,
            "gt_masks": mask,
        }

    def _load_map(
        self, episode: CoreContainer, world_tf_ego: np.ndarray
    ) -> dict[str, Any]:
        town = episode.episode_meta.region[0]
        layers, rasterization = self.map_extractor.get_map_segmentation(
            map_name=town,
            world_tf_ego=world_tf_ego,
            bev_range=self.bev_range[3],
            bev_shape=(self.map_resolution, self.map_resolution),
        )

        # build a 3-channel map consisting of driveable, lane markings and stop line
        driveable = rasterization[0].clip(0.0, 1.0)
        lane_markings = (rasterization[1] + rasterization[2] + rasterization[3]).clip(
            0.0, 1.0
        )
        stop_line = rasterization[4].clip(0.0, 1.0)

        map_segmentation = np.stack(
            [driveable, lane_markings, stop_line], axis=0
        )  # Shape: (3, H, W)
        return dict(map_segmentation=torch.tensor(map_segmentation, dtype=self.dtype))

    def _load_planning(
        self,
        episode: CoreContainer,
        frame_idx: int,
        world_tf_ego: np.ndarray,
        ego_tf_world: np.ndarray,
    ) -> dict[str, Any]:
        planning = {}

        ego_future = self._get_ego_future(episode, frame_idx, ego_tf_world)
        planning.update(ego_future)

        bev_waypoints = self._get_bev_waypoints(
            episode, frame_idx, world_tf_ego, ego_tf_world
        )
        planning.update(bev_waypoints)

        planning_command = self._get_command(
            episode, frame_idx, ego_tf_world=ego_tf_world
        )
        planning["command"] = np.array([planning_command], dtype=np.int64)
        target_point = self._get_target_point(episode, frame_idx, ego_tf_world)
        planning.update(target_point)

        return planning

    def _get_ego_future(
        self, episode: CoreContainer, frame_idx: int, ego_tf_world: np.ndarray
    ) -> dict[str, Any]:
        planning_traj = np.zeros((self.planning_frames, 4), dtype=np.float32)
        planning_speed = np.zeros((self.planning_frames,), dtype=np.float32)
        planning_mask = np.zeros((self.planning_frames,), dtype=bool)

        ws = episode.world_state
        ref_tf_futures = ego_tf_world[np.newaxis, :, :] @ ws.transform[0, :, 0]

        step_size = self.dataset_frame_rate // self.planning_frame_rate
        for i, index_offset in enumerate(
            range(step_size, (self.planning_frames + 1) * step_size, step_size)
        ):
            idx = frame_idx + index_offset

            if idx >= ws.shape[1]:
                break

            # trajectory
            ref_tf_future = ref_tf_futures[idx]
            sin_yaw = ref_tf_future[1, 0]
            cos_yaw = ref_tf_future[0, 0]
            planning_traj[i, :2] = ref_tf_future[0:2, 3]
            planning_traj[i, 2] = sin_yaw
            planning_traj[i, 3] = cos_yaw

            # speed
            speed = ws.dynamics[0, idx, 0, 0]  # speed in m/s
            planning_speed[i] = speed

            # mask
            planning_mask[i] = True

        return {
            "planning_traj": planning_traj,
            "planning_speed": planning_speed,
            "planning_mask": planning_mask,
        }

    def _get_bev_waypoints(
        self,
        episode: CoreContainer,
        frame_idx: int,
        world_tf_ego: np.ndarray,
        ego_tf_world: np.ndarray,
    ) -> dict[str, Any]:
        def requires_replanning(current_command, next_command):
            """Returns True iff the current command and the next command are different and not trivially compatible."""

            # convert commands back to carla range
            current_command += 1
            next_command += 1
            assert current_command in range(1, 7)
            assert next_command in range(1, 7)

            if not self.require_replanning:
                # replanning is disabled
                return False
            if current_command == next_command:
                return False
            if current_command == 4:
                # follow lane is not compatible with (left, right, change-lane-{left,right})
                return next_command in (1, 2, 5, 6)
            # all other commands do not require a replanning
            return False

        current_command = self._get_command(episode, frame_idx)
        search_ind = frame_idx + 1
        cum_dist = 0
        waypoints = []
        last_marker = world_tf_ego[:3, 3]
        last_wp = last_marker[:3]

        while len(waypoints) < self.num_bev_waypoints:
            if search_ind >= episode.world_state.shape[1]:
                break
            future_command = self._get_command(episode, search_ind)
            if requires_replanning(current_command, future_command):
                break
            world_tf_next = episode.world_state.transform[0, search_ind, 0]
            next_marker = world_tf_next[:3, 3]

            # interpolate by factor of 8
            between_markers = [
                (15 * last_marker + 1 * next_marker) / 16,
                (14 * last_marker + 2 * next_marker) / 16,
                (13 * last_marker + 3 * next_marker) / 16,
                (12 * last_marker + 4 * next_marker) / 16,
                (11 * last_marker + 5 * next_marker) / 16,
                (10 * last_marker + 6 * next_marker) / 16,
                (9 * last_marker + 7 * next_marker) / 16,
                (8 * last_marker + 8 * next_marker) / 16,
                (7 * last_marker + 9 * next_marker) / 16,
                (6 * last_marker + 10 * next_marker) / 16,
                (5 * last_marker + 11 * next_marker) / 16,
                (4 * last_marker + 12 * next_marker) / 16,
                (3 * last_marker + 13 * next_marker) / 16,
                (2 * last_marker + 14 * next_marker) / 16,
                (1 * last_marker + 15 * next_marker) / 16,
                (0 * last_marker + 16 * next_marker) / 16,
            ]
            for marker in between_markers:
                this_wp = marker
                dist = np.linalg.norm(this_wp - last_wp)
                if cum_dist + dist >= (len(waypoints) + 1) * self.bev_waypoint_distance:
                    # add this waypoint
                    cum_dist += dist
                    last_wp = this_wp
                    waypoints.append(marker)
                if len(waypoints) >= self.num_bev_waypoints:
                    break

            last_marker = next_marker
            search_ind += 1

        # convert bev waypoints to ego T=0
        bev_waypoints = np.zeros((self.num_bev_waypoints, 2), dtype=np.float32)
        bev_waypoints_mask = np.zeros((self.num_bev_waypoints,), dtype=bool)
        for i, waypoint in enumerate(waypoints):
            wp_ego = ego_tf_world @ np.append(waypoint, 1)
            bev_waypoints[i, :] = wp_ego[:2]
            bev_waypoints_mask[i] = 1

        return {
            "bev_waypoints": bev_waypoints,
            "bev_waypoints_mask": bev_waypoints_mask,
        }

    def _get_command(
        self,
        episode: CoreContainer,
        frame_idx: int,
        ego_tf_world: np.ndarray | None = None,
    ) -> dict[str, Any]:
        cmd_index = max(
            0, episode.routing_information.navigation_goal[0, frame_idx] - 1
        )
        command = episode.routing_information.route_commands[0][cmd_index] - 1

        if self.fix_lane_change_commands and ego_tf_world is not None:
            # check if previous is a lane-change command
            prev_cmd_index = max(0, cmd_index - 1)
            prev_command = (
                episode.routing_information.route_commands[0][prev_cmd_index] - 1
            )
            if prev_command in (4, 5) and command != prev_command:
                # the current command ends the lane change
                # we check if the car has indeed reached this point
                lane_change_end_point = episode.routing_information.target_route[0][
                    cmd_index
                ]
                lane_change_end_point = np.append(lane_change_end_point, 1).reshape(
                    -1, 1
                )

                lane_change_end_point_ego = ego_tf_world @ lane_change_end_point
                if lane_change_end_point_ego[0] > 0:
                    # the end point is ahead of the ego, i.e., the lane change is not complete
                    # we overwrite the conditioning with the previous command
                    command = prev_command

        return command

    def _get_target_point(
        self, episode: CoreContainer, frame_idx: int, ego_tf_world: np.ndarray
    ) -> dict[str, Any]:
        target_index = episode.routing_information.navigation_goal[0, frame_idx]
        target_point_world = episode.routing_information.target_route[0, target_index]
        target_point_ego = ego_tf_world @ np.append(target_point_world, 1)

        return {
            "target_point": target_point_ego[:3],
        }

    def _get_images(self, image_paths):
        # select normalization based on the backbone type
        if self.backbone_type in ("ResNet", "DINO"):
            normalization = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )  # ImageNet normalization
        elif self.backbone_type in ("RADIO",):
            # no additional normalization needed
            normalization = transforms.Identity()

        # select augmentation based on the backbone type
        if self.backbone_type in ("ResNet",):
            augmentation = transforms.RandomPhotometricDistort()
        else:
            # no augmentation needed for foundation models
            augmentation = transforms.Identity()

        # check if the image cache exists
        if all(
            os.path.isfile(osp.join(self.img_cache, cam_path))
            for cam_path in image_paths
        ):
            # simplified image pipeline
            img_pipeline = transforms.Compose(
                [
                    transforms.ToImage(),  # PIL->tensor (run first for best performance)
                    transforms.ConvertImageDtype(self.dtype),
                    augmentation,
                    normalization,
                ]
            )

            # load from cache
            def load_and_process(image_path):
                local_path = os.path.join(self.img_cache, image_path)
                img = Image.open(local_path)
                img = img_pipeline(img)
                return img

        else:
            # full image pipeline
            img_pipeline_1 = transforms.Compose(
                [
                    transforms.ToImage(),  # PIL->tensor (run first for best performance)
                    transforms.Resize(
                        self.img_resolution,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),  # run second to minimize memory usage
                ]
            )
            img_pipeline_2 = transforms.Compose(
                [
                    transforms.ConvertImageDtype(self.dtype),
                    augmentation,
                    normalization,
                ]
            )

            # download images and save to cache
            def load_and_process(image_path):
                remote_path = os.path.join(self.episode_base, image_path)
                local_path = os.path.join(self.img_cache, image_path)
                img = Image.open(remote_path)
                small_image = img_pipeline_1(img)  # resize and crop
                save_image = transforms.ToPILImage()(small_image)
                local_base_path = os.path.dirname(local_path)
                Path(local_base_path).mkdir(
                    parents=True, exist_ok=True
                )  # ensure directory exists
                save_image.save(local_path)
                nn_image = img_pipeline_2(small_image)  # normalize and augment
                return nn_image

        with ThreadPoolExecutor(max_workers=len(image_paths)) as executor:
            images = list(executor.map(load_and_process, image_paths))

        return torch.stack(images, dim=0)  # shape: (N, C, H, W)

    def _extract_label(self, ws_or_tce, frame_idx) -> np.ndarray:
        if isinstance(ws_or_tce, WorldState):
            category = ws_or_tce.category[0, frame_idx]
            _category_map = ws_or_tce.category_map
        elif isinstance(ws_or_tce, TrafficControlElements):
            category = ws_or_tce.state[0, frame_idx]
            _category_map = ws_or_tce.state_map
        else:
            raise ValueError(f"Unsupported type: {type(ws_or_tce)}.")

        category_map = {v: k for k, v in _category_map.items()}  # invert the map
        category_map[0] = "car"  # masked out labels will be handled outside this method

        category_string = [category_map[c] for c in category]
        labels = np.array(
            [
                self.class2id[self.class_mapping[c]] if c in self.class_mapping else 0
                for c in category_string
            ],
            dtype=np.int64,
        )
        valid_labels = np.array(
            [c in self.class_mapping for c in category_string], dtype=bool
        )
        return labels, valid_labels
