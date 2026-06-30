import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from agents.navigation.local_planner import RoadOption
from bevad_sim.data_interface.action import Action
from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.simulation.agent.base_agent import BaseAgent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import load_checkpoint

import bevad.modules
from bevad.agent.pid_controller import build_controller

MINIMUM_LANE_CHANGE_CMD_LENGTH = 8  # meters


def downsample_route(route, sample_factor):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # At the beginning
        if prev_option is None:
            ids_to_sample.append(i)
            dist = 0

        # Lane changing
        elif curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When entering or exitting intersections
        elif prev_option != curr_option and prev_option not in (
            RoadOption.CHANGELANELEFT,
            RoadOption.CHANGELANERIGHT,
        ):
            ids_to_sample.append(i)
            dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i - 1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


class InferenceModel(nn.Module):
    pass


class BevadAgent(BaseAgent):
    def __init__(self, config):
        self.cfg_str = config["cfg"]
        self.device = "cuda"
        self.dtype = torch.float16

        if "compile_model" in config and config["compile_model"]:
            self.compile_model = True
        else:
            self.compile_model = False

        self.LOG_ATTENTION_WEIGHTS = os.environ.get("LOG_ATTENTION_WEIGHTS", "0") == "1"

        self.model_frequency = None  # to be set in derived class
        self.progress = 0.0
        self.progress_of_last_lanechange = 0.0
        self.last_command_idx = 0

    @torch.no_grad()
    def run_step(self, _observations: CoreContainer):
        """Run on observations.

        Receives a list of observations, runs the agents aggregator function,
        executes the actual model and returns an action

        Returns
        -------
            tuple: (acceleration, steering angle).
        """

        frame_idx = 0
        world_state = _observations.world_state
        odometry = _observations.odometry

        ###############################################

        # select normalization based on the backbone type
        if self.cfg.image_backbone in ("ResNet", "DINO"):
            normalization = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )  # ImageNet normalization
        elif self.cfg.image_backbone in ("RADIO",):
            # no additional normalization needed
            normalization = transforms.Identity()
        else:
            raise ValueError(
                "Image normalization type not supported: {}".format(
                    self.cfg.image_backbone
                )
            )

        # select resize if image size does not match
        if (
            self.cfg.image_resolution[0]
            != _observations.camera_observations["rgb"].height
            or self.cfg.image_resolution[1]
            != _observations.camera_observations["rgb"].width
        ):
            resize = transforms.Resize(
                self.cfg.image_resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        else:
            resize = transforms.Identity()

        img_pipeline = transforms.Compose(
            [
                transforms.ToImage(),  # PIL->tensor (run first for best performance)
                resize,  # run second to minimize memory usage
                transforms.ConvertImageDtype(self.dtype),
                normalization,
            ]
        )

        # images
        cam_names = [
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ]
        cam_idx = [
            _observations.camera_observations["rgb"].sensor_names.index(cam_name)
            for cam_name in cam_names
        ]
        images = [
            img_pipeline(
                torch.tensor(
                    _observations.camera_observations["rgb"].data[0, 0, i],
                    device=self.device,
                )
            )
            for i in cam_idx
        ]
        img = torch.stack(images, dim=0).unsqueeze(0)

        # camera transformations
        img_tf_lidar = []
        for i in cam_idx:
            # scale intrinsics
            img_tf_cam = np.eye(4)
            img_tf_cam[:3, :3] = _observations.camera_observations["rgb"].intrinsics[
                0, 0, i
            ]
            if (
                self.cfg.image_resolution[0]
                == _observations.camera_observations["rgb"].height
            ):
                # no intrinisic adaption needed
                pass
            elif (
                self.cfg.image_resolution == (448, 800)
                and _observations.camera_observations["rgb"].height == 900
            ):
                # TODO: fix me this is a hack to compensate for the image resize
                img_tf_cam[0, 2] /= 2
                img_tf_cam[1, 2] /= 2
                img_tf_cam[0, 0] /= 2
                img_tf_cam[1, 1] /= 2
            elif self.cfg.image_resolution == (896, 1600):
                # no intrinisic adaption needed
                pass
            else:
                raise ValueError(
                    f"Unsupported image resolution: {self.cfg.image_resolution}. Supported resolutions are (448, 800) and (896, 1600)."
                )

            cam_tf_vehicle = _observations.camera_observations["rgb"].extrinsics[
                0, 0, i
            ]
            cam_tf_lidar = cam_tf_vehicle @ self.lidar2ego
            img_tf_lidar.append(img_tf_cam @ cam_tf_lidar)
        img_tf_lidar = np.stack(img_tf_lidar, axis=0)
        img_tf_lidar = torch.tensor(
            img_tf_lidar, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        # localization (TODO: only valid for static models)
        localization = torch.zeros(1, 8, device=self.device, dtype=self.dtype)

        # dynamics
        dynamics = np.zeros(9, dtype=np.float32)
        dynamics[0] = world_state.dynamics[0, frame_idx, 0, 0]
        # TODO: handle NaN values in dynamics
        acceleration = odometry.acceleration[0, frame_idx]
        if np.any(np.isnan(acceleration)) or np.any(np.abs(acceleration) > 200):
            acceleration = np.zeros_like(acceleration)
        dynamics[3:6] = acceleration
        dynamics[6:9] = odometry.angular_velocity[0, frame_idx]
        dynamics = torch.tensor(dynamics, device=self.device, dtype=self.dtype)[
            None, ...
        ]

        # command & speed
        cmd_index = max(
            0, _observations.routing_information.navigation_goal[0, frame_idx] - 1
        )
        command = _observations.routing_information.route_commands[0][cmd_index] - 1
        current_speed = torch.tensor(
            _observations.world_state.dynamics[0, 0, 0, 0:1], device=self.device
        ).float()  # fake the speed sensor, TODO: add noise

        # check if previous command was a lane change and if we have moved enough to update the command
        last_command_idx = max(0, cmd_index - 1)
        if (
            last_cmd := _observations.routing_information.route_commands[0][
                last_command_idx
            ]
        ) in (
            5,
            6,
        ):
            # last command was a lane change
            progress_since_last_lanechange = (
                self.progress - self.progress_of_last_lanechange
            )
            if progress_since_last_lanechange < MINIMUM_LANE_CHANGE_CMD_LENGTH:
                # not enough progress yet, keep the previous command
                print(
                    "Extend lane change command for ",
                    progress_since_last_lanechange,
                    "m",
                )
                command = last_cmd - 1

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output_data_batch, _ = self.model(
                img=img,
                img_tf_lidar=img_tf_lidar,
                localization=localization,
                dynamics=dynamics,
                command=torch.tensor(
                    [[command]], dtype=torch.int64, device=self.device
                ),
                current_speed=current_speed,
                model_frequency=self.model_frequency,
                episode_id=["eval"],
                frame_id=torch.tensor(
                    [self.step], dtype=torch.int64, device=self.device
                ),
            )

        ###############################################

        # determine mode: trajectory vs. disentangled
        planning_output = output_data_batch["planning"]
        pred_trajectory = None
        pred_bev_waypoints = None
        target_speed = None
        if "pred_trajectory" in planning_output:
            # trajectory mode
            pred_trajectory = planning_output["pred_trajectory"][0].cpu().numpy()

            if (
                "planning_mask" in planning_output
                and planning_output["planning_mask"] is not None
            ):
                planning_mask = planning_output["planning_mask"][0].cpu().numpy()
                pred_trajectory = pred_trajectory[planning_mask]
            _observations.step_meta.info[0][0]["pred_trajectory"] = pred_trajectory
        if (
            "pred_bev_waypoints" in planning_output
            and planning_output["pred_bev_waypoints"] is not None
        ):
            # disentangled mode
            pred_bev_waypoints = planning_output["pred_bev_waypoints"][0].cpu().numpy()
            _observations.step_meta.info[0][0]["pred_bev_waypoints"] = (
                pred_bev_waypoints
            )
        if "speed" in planning_output:
            target_speed = planning_output["speed"][0].cpu().numpy()
            _observations.step_meta.info[0][0]["speed"] = target_speed

        if "roll_out_pos" in output_data_batch:
            roll_out_pos = output_data_batch["roll_out_pos"].cpu().numpy()
            roll_out_yaw = output_data_batch["roll_out_yaw"].cpu().numpy()

            _observations.step_meta.info[0][0]["roll_out_pos"] = roll_out_pos
            _observations.step_meta.info[0][0]["roll_out_yaw"] = roll_out_yaw

        if "detection" in output_data_batch:
            scores = output_data_batch["detection"]["all_cls_scores"][-1][0]
            boxes = output_data_batch["detection"]["all_bbox_preds"][-1][0]

            _observations.step_meta.info[0][0]["detection_scores"] = (
                scores.cpu().numpy()
            )
            _observations.step_meta.info[0][0]["detection_boxes"] = boxes.cpu().numpy()

        speed = current_speed[0].item()
        # run PID controller
        steer_traj, throttle_traj, brake_traj, metadata_traj = (
            self.pidcontroller.control_pid(
                bev_waypoints=pred_bev_waypoints,
                trajectory=pred_trajectory,
                target_speed=target_speed,
                current_speed=speed,
                frame_rate=self.cfg.planning_frame_rate,
            )
        )
        print(f"v={speed:02.1f}", f"CMD={command}")
        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0

        if "aim_point" in metadata_traj:
            _observations.step_meta.info[0][0]["aim_point"] = metadata_traj["aim_point"]

        simulator_control = {
            "throttle": throttle_traj,
            "steer": steer_traj,
            "brake": brake_traj,
        }

        if _observations.action is None:
            _observations.action = Action.create_empty()
        _observations.action.simulator_control = [[simulator_control]]

        self.step += 1

        # update progress
        self.progress += speed * (1 / 20)

        # detect lane change triggers
        if command + 1 in (5, 6) and cmd_index != self.last_command_idx:
            self.progress_of_last_lanechange = self.progress

        # update last command index
        self.last_command_idx = cmd_index

        return _observations

    def end_episode(self):
        """
        End the current episode.
        This method is intended to be overridden by subclasses to perform any
        necessary cleanup or finalization at the end of an episode.
        If the agent creates any results (such as some video), the result is expected to be returned here.
        """

        return None

    def setup(self, path_to_conf_file):
        self.pidcontroller = build_controller()
        self.config_path = path_to_conf_file.split("+")[0]
        self.ckpt_path = path_to_conf_file.split("+")[1]

        self.step = -1
        cfg = Config.fromfile(self.config_path)
        self.cfg = cfg

        model = build_model(
            cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
        )
        inference_model = InferenceModel()
        inference_model.model = model
        checkpoint = load_checkpoint(
            inference_model, self.ckpt_path, map_location="cpu", strict=True
        )
        model = inference_model.model
        if self.compile_model:
            self.model = torch.compile(model)
        else:
            self.model = model
        self.model.cuda()
        self.model.eval()

        self.lidar2ego = np.array(
            [
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.step = 0


class BevadClosedLoopAgent(BevadAgent):
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1])
            for x in ds_ids
        ]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

    def start_episode(self, env: Optional[Any] = None):
        """
        Starts a new episode for the agent.
        This method should be called at the beginning of each episode to initialize
        any necessary parameters or states for the agent and to reset the agent in between runs in the same environment.
        """

        simulator = env.unwrapped
        self.set_global_plan(
            simulator.route_scenario.gps_route, simulator.route_scenario.route
        )
        self.setup(self.cfg_str)

        self.model_frequency = 20


def get_class():
    return BevadClosedLoopAgent
