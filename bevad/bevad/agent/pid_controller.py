from collections import deque

import numpy as np
from scipy.interpolate import PchipInterpolator


class PID(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class PIDController(object):
    def __init__(
        self,
        turn_KP=0.75,
        turn_KI=0.75,
        turn_KD=0.3,
        turn_n=40,
        speed_KP=5.0,
        speed_KI=0.5,
        speed_KD=1.0,
        speed_n=40,
        max_throttle=0.75,
        brake_speed=0.4,
        brake_ratio=1.1,
        clip_delta=0.25,
        aim_dist=4.0,
        angle_thresh=0.3,
        dist_thresh=10,
    ):
        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh

    def control_pid(self, waypoints, speed, target):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            speed (tensor): speedometer input
        """

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += (
                np.linalg.norm(waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs
            )

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (
            np.abs(angle_target - angle_last) > self.angle_thresh
            and target[1] < self.dist_thresh
        )
        if use_target_to_aim:
            print("Correct angle.")
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        brake = (
            desired_speed < self.brake_speed
            or (speed / desired_speed) > self.brake_ratio
        )

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 1.0)  # self.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            "speed": float(speed.astype(np.float64)),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "wp_4": tuple(waypoints[3].astype(np.float64)),
            "wp_3": tuple(waypoints[2].astype(np.float64)),
            "wp_2": tuple(waypoints[1].astype(np.float64)),
            "wp_1": tuple(waypoints[0].astype(np.float64)),
            "aim": tuple(aim.astype(np.float64)),
            "target": tuple(target.astype(np.float64)),
            "desired_speed": float(desired_speed.astype(np.float64)),
            "angle": float(angle.astype(np.float64)),
            "angle_last": float(angle_last.astype(np.float64)),
            "angle_target": float(angle_target.astype(np.float64)),
            "angle_final": float(angle_final.astype(np.float64)),
            "delta": float(delta.astype(np.float64)),
        }

        return steer, throttle, float(brake), metadata


class LateralPIDController(object):
    """
    PID controller
    """

    def __init__(
        self,
        k_p=3.118357247806046,
        k_d=1.3782508892109167,
        k_i=0.6406067986034124,
        speed_scale=0.9755321901954155,
        speed_offset=1.9152884533402488,
        default_lookahead=24,
        speed_threshold=23.150102938235136,
        n=6,
        inference_mode=False,
    ):
        self.k_p = 1.8  # k_p
        self.k_d = k_d
        self.k_i = k_i
        self.speed_scale = speed_scale
        self.speed_offset = speed_offset
        self.default_lookahead = default_lookahead
        self.speed_threshold = speed_threshold
        self.n = n
        self.inference_mode = inference_mode  # False when used in the expert, True when used with a trained model

        self._saved_window = []
        self._window = []

    def step(self, route_np, current_speed):
        """Run turning PID controller.

        Args:
            route_np (_type_): Path waypoints in left-handed ego coordinates.
            current_speed (_type_): Current speed of the vehicle in m/s.

        Returns:
            _type_: Steering angle command.
        """

        # used at leaderboard
        current_speed = current_speed * 3.6
        n_lookahead = int(
            min(
                np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105),
                route_np.shape[0] - 1,
            )
        )

        n_lookahead = min(n_lookahead, len(route_np) - 1)
        desired_heading_vec = route_np[n_lookahead]
        aim_point = np.copy(desired_heading_vec)
        aim_point[1] *= -1

        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = (yaw_path) % (2 * np.pi)
        heading_error = (
            heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        )

        # the scaling doesn't deserve any specific purpose but is a leftover from a previous less efficient implementation,
        # on which we optimized the parameters
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(heading_error)
        self._window = self._window[-self.n :]

        derivative = (
            0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        )
        integral = np.mean(self._window)

        steering = np.clip(
            self.k_p * heading_error + self.k_d * derivative + self.k_i * integral,
            -1.0,
            1.0,
        ).item()

        return steering, aim_point


DEFAULT_PID_CFG = dict(
    aim_distance_fast=3.0,
    aim_distance_slow=2.25,
    aim_distance_threshold=5.5,
    turn_KP=1.3,
    turn_KI=0.75,
    turn_KD=0.3,
    turn_n=10,  # buffer size
    speed_KP=1.75,
    speed_KI=0.5,
    speed_KD=3.0,
    speed_n=20,  # buffer size
    brake_speed=0.4,  # to prevent creeping 0.4,  # desired speed below which brake is triggered
    brake_ratio=1.1,  # ratio of speed to desired speed at which brake is triggered
    clip_delta=1.0,  # maximum change in speed input to logitudinal controller
    clip_throttle=1.0,
    # Only for advanced PID controller
    angle_thresh=0.3,  # outlier control detection angle
    dist_thresh=10,  # target point y-distance for outlier filtering
    is_stuck_speed=0.5,  # speed at which the agent is assumed to be stuck
    speed_weight=0.05,
    value_weight=0.001,
    features_weight=0.05,
)


class CarlaPIDControl(object):
    def __init__(
        self,
        turn_controller,
        speed_controller,
        stuck_threshold=5,
    ) -> None:
        self.turn_controller = turn_controller
        self.speed_controller = speed_controller
        self.stuck_detector = 0
        self.stuck_threshold = stuck_threshold

    def control_pid(
        self,
        *,
        bev_waypoints=None,
        trajectory=None,
        target_speed=None,
        current_speed,
        frame_rate,
    ):
        if target_speed is not None:
            desired_speed = float(target_speed[0])
        elif trajectory is not None:
            # calculate desired speed from trajectory
            desired_speed = self._derive_speed_from_trajectory(trajectory, frame_rate)
        else:
            raise ValueError("No longitudinal target provided.")

        if bev_waypoints is not None:
            # calculate aim point in disentangled mode
            aim_world = self._determine_aim_point(bev_waypoints, current_speed)
        elif trajectory is not None:
            # calculate aim point in trajectory mode
            aim_world = self._determine_aim_point(trajectory, current_speed)
        else:
            raise ValueError("No lateral target provided.")

        if bev_waypoints is not None:
            bev_waypoints = bev_waypoints.copy()
            # convert to left-handed coordinates
            bev_waypoints *= np.array([1, -1])

        if trajectory is not None:
            trajectory = trajectory.copy()
            # convert to left-handed coordinates
            trajectory *= np.array([1, -1])

        brake, throttle, steer, aim_ego = self.run_pid_control_for_targets(
            speed=current_speed,
            desired_speed=desired_speed,
            aim_world=aim_world,
            ego_T_world=np.eye(4),
            bev_waypoints=bev_waypoints,
            trajectory=trajectory,
        )
        return steer, throttle, brake, dict(aim_point=aim_ego)

    def _derive_speed_from_trajectory(self, trajectory, frame_rate):
        v, a = self._derive_trajecotry(trajectory, delta_t=1 / frame_rate)

        # by default, use the speed at 0.5s-1s sec ahead
        v05 = v[int(frame_rate * 0.5) - 1]  # v(0.5 sec)
        v10 = v[int(frame_rate * 1) - 1]  # v(1.0 sec)
        desired_speed = (v05 + v10) / 2.0

        return desired_speed

    def _determine_aim_point(self, points, current_speed):
        """Choose an aim point from the predicted path based on the current speed."""

        # clip desired speed to reasonable values
        current_speed = np.clip(current_speed, 0, 12)
        # we find aim distances between 2m and 8m reasonable, so we linearly map the desired speed to that range
        aim_distance = 2 + 0.5 * current_speed

        # We follow the waypoint that is at least a certain distance away
        aim_index = points.shape[0] - 1
        for index, predicted_waypoint in enumerate(points):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim_ego = points[aim_index]
        aim_ego_hom = np.array([aim_ego[0], aim_ego[1], 0, 1], dtype=aim_ego.dtype)
        aim_world = np.eye(4) @ aim_ego_hom

        return aim_world

    def run_pid_control_for_targets(
        self,
        speed,
        desired_speed,
        aim_world,
        ego_T_world,
        bev_waypoints=None,
        trajectory=None,
    ):
        brake = (desired_speed < DEFAULT_PID_CFG["brake_speed"]) or (
            (speed / desired_speed) > DEFAULT_PID_CFG["brake_ratio"]
        )

        delta = np.clip(desired_speed - speed, 0.0, DEFAULT_PID_CFG["clip_delta"])
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, DEFAULT_PID_CFG["clip_throttle"])
        throttle = throttle if not brake else 0.0

        aim_ego = ego_T_world @ aim_world
        # LHS RHS to align with carla ISO
        # aim_ego = aim_ego * np.array([1, -1, 1, 1])
        aim_ego = aim_ego[:2]  # controller only uses x,y
        angle = np.degrees(np.arctan2(-aim_ego[1], aim_ego[0])) / 90.0
        # angle = np.degrees(np.pi / 2 - np.arctan2(aim_ego[1], aim_ego[0])) / 90

        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate
            # in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        # steer = self.turn_controller.step(angle)
        if bev_waypoints is not None:
            route_interp = self.interpolate_waypoints(bev_waypoints.squeeze())
            steer, aim_ego = self.turn_controller.step(route_interp, speed)
        else:
            route_interp = self.interpolate_trajectory(trajectory.squeeze())
            steer, aim_ego = self.turn_controller.step(route_interp, speed)
        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]

        return float(brake), throttle, steer, aim_ego

    def _derive_trajecotry(self, trajectory, delta_t):
        trajectory = np.concatenate((np.zeros((1, 2)), trajectory), axis=0)
        delta_s = trajectory[1:, :] - trajectory[:-1, :]
        delta_s = np.sqrt(delta_s[:, 0] ** 2 + delta_s[:, 1] ** 2)
        delta_s = np.concatenate((np.zeros((1,)), delta_s), axis=0)
        s = np.cumsum(delta_s)

        # compute speed using central differences v = (s(t+1) - s(t-1)) / (2 * delta_t)
        v = (s[2:] - s[:-2]) / (2 * delta_t)  # v(1), v(2), ..., v(n-1)

        # compute acceleration using central differences a = (s(t+1) - 2 * s(t) + s(t-1)) / delta_t^2
        a = (s[2:] - 2 * s[1:-1] + s[:-2]) / (delta_t**2)  # a(1), a(2), ..., a(n-1)

        return v, a

    def interpolate_waypoints(self, waypoints):
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        dists += (
            np.arange(0, len(dists)) * 1e-4
        )  # Prevents dists not being strictly increasing

        interp = PchipInterpolator(dists, waypoints, axis=0)

        x = np.arange(0.1, dists[-1], 0.1)

        interp_points = interp(x)

        # There is a possibility that all points are at 0, meaning there is no point distanced 0.1
        # In this case we output the last (assumed to be furthest) waypoint.
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]

        return interp_points

    def interpolate_trajectory(self, waypoints):
        waypoints = waypoints.copy().astype(np.float32)
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(waypoints - shift, axis=1).astype(np.float32)
        dists = np.cumsum(dists)
        dists += (
            np.arange(0, len(dists)) * 1e-2
        )  # Prevents dists not being strictly increasing

        interp = PchipInterpolator(dists, waypoints, axis=0)
        x = np.arange(0.1, dists[-1], 0.1)
        interp_points = interp(x)

        # There is a possibility that all points are at 0, meaning there is no point distanced 0.1
        # In this case we output the last (assumed to be furthest) waypoint.
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]

        return interp_points


def build_controller():
    turn_controller = PID(
        K_P=DEFAULT_PID_CFG["turn_KP"],
        K_I=DEFAULT_PID_CFG["turn_KI"],
        K_D=DEFAULT_PID_CFG["turn_KD"],
        n=DEFAULT_PID_CFG["turn_n"],
    )
    speed_controller = PID(
        K_P=DEFAULT_PID_CFG["speed_KP"],
        K_I=DEFAULT_PID_CFG["speed_KI"],
        K_D=DEFAULT_PID_CFG["speed_KD"],
        n=DEFAULT_PID_CFG["speed_n"],
    )
    turn_controller = LateralPIDController()

    return CarlaPIDControl(
        turn_controller=turn_controller, speed_controller=speed_controller
    )
