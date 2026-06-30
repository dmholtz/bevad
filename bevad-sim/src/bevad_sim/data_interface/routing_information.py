from __future__ import annotations

import math
from dataclasses import Field, dataclass
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import shapely.geometry
import torch
from typing_extensions import Self

from bevad_sim.data_interface.base_entity import BaseEntity
from bevad_sim.data_interface.episode_map import EpisodeMap
from bevad_sim.data_interface.world_state import TransformsOperations, transform_2d_seq

if TYPE_CHECKING:
    from bevad_sim.data_interface.core_container import CoreContainer
    from bevad_sim.data_interface.data_types import HighLevelCommands


@dataclass
class RoutingInformation(BaseEntity):
    """Encapsulates episode-level and step-level routing information for navigation tasks.
    This class stores information such as the target lane sequence, tactical search space,
    target route, route commands, and navigation goals. It supports both numpy arrays and
    PyTorch tensors for flexible integration with different ML frameworks.

    Attributes:
        target_lane_sequence (np.ndarray | None):
            The sequence of target lanes for the episode. Shape: (batch, ...).
        tactical_search_space (np.ndarray | None):
            The tactical search space for the episode. Shape: (batch, ...).
        target_route (np.ndarray | torch.Tensor | None):
            The target route as a sequence of (x, y) coordinates. Shape: (batch, N, 2).
        route_commands (list[list[HighLevelCommands]] | None):
            List of high-level route commands for each batch.
        navigation_goal (np.ndarray | torch.Tensor | None):
            route commands. Shape: (batch, time).

    Properties:
        shape (tuple):
            Returns the shape of the navigation_goal if set, otherwise attempts to infer shape from other fields.

    Methods:
        transform_to(matrix, target_ri):
            Transforms the target_route using the provided transformation matrix and assigns it to target_ri.

    Note:
        - The class is designed to be compatible with both numpy and PyTorch tensors.
        - Some fields may be None if not set for a particular episode or step.
    """

    # episode-level
    target_lane_sequence: (
        np.ndarray | None
    )  # field(default_factory=lambda: np.zeros((1, 1), dtype=np.int32))  # Batch, no time
    tactical_search_space: np.ndarray | None  # field(
    #    default_factory=lambda: np.zeros((1, 1), dtype=np.int32)
    # )  # Batch, no time
    target_route: np.ndarray | torch.Tensor | None  # field(
    #   default_factory=lambda: np.zeros((1, N, 2), dtype=np.float32)
    # )  # Batch, not time, (x, y)
    route_commands: list[list[HighLevelCommands]] | None  # field(default_factory=list)  # batch, not time

    # step-level
    navigation_goal: np.ndarray | torch.Tensor | None  # field(
    # default_factory=lambda: np.zeros((1, 2), dtype=np.int32)
    # )  # Batch, time

    @property
    def dimensionality(self) -> int:
        return 2

    @property
    def t_dim(self) -> None:
        """Returns the size of the time dimension. Returns none since object has no time dimension."""
        return None

    @property
    def n_dim(self) -> None:
        """Returns the size of the element dimension. Returns none since object has no element dimension."""
        return None

    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        self._check_array_dim("target_lane_sequence", 1, None, ignore_list)
        self._check_array_dim("tactical_search_space", 1, None, ignore_list)
        self._check_array_dim("target_route", 1, (2,), ignore_list)
        self._check_list_dim("route_commands", 1, None, ignore_list)
        self._check_array_dim("navigation_goal", 2, None, ignore_list)

    @classmethod
    def aggregated_time(cls, batch: list[Self], use_custom_batching: list | None = None) -> Self:
        """
        Combine the static and dynamic attributes into a time-batched RoutingInformation object.
        """
        # dynamic
        if batch[-1].navigation_goal is not None:
            goals = [b.navigation_goal for b in batch if b.navigation_goal is not None]
            navigation_goal = BaseEntity._concat_tensors(goals, dim=1)
            is_valid = BaseEntity._concat_tensors([b.is_valid for b in batch], dim=1)
        else:
            navigation_goal = None

        return cls(
            is_valid=is_valid,
            target_lane_sequence=batch[-1].target_lane_sequence,
            tactical_search_space=batch[-1].tactical_search_space,
            target_route=batch[-1].target_route,
            route_commands=batch[-1].route_commands,
            navigation_goal=navigation_goal,
        )

    def __getitem__(self, idx: int | Tuple, use_custom_slicing: List | None = None) -> Self:
        ### RoutingInformation doesn't have time dimension except for the navigation_goal. Remove it from idx.
        batch_only_idx = (idx[0],) if isinstance(idx, tuple) else idx
        res = super().__getitem__(batch_only_idx, ["navigation_goal"])
        if self.navigation_goal is not None:
            res.navigation_goal = self.navigation_goal[idx]  ## navigation_goal has time dimension, use original idx.
        return res

    def transform_to(self, matrix):
        """
        Transforms the target_route attribute using the provided matrix.

        Args:
            matrix: The transformation matrix to apply to the target_route.

        Returns:
            new RoutingInformation object with copy of all members except target rout

        """
        assert self.shape[0] == 1, "For now transforming batched data is not supported! "

        t_tr = np.concatenate((self.target_route[..., :2], np.zeros_like(self.target_route[..., :2])), axis=-1)
        t_tr[..., 3] = 1.0

        new_tr = t_tr @ np.transpose(matrix[0])
        # TODO: What about navigation goal??
        # navigation_goal

        return RoutingInformation(
            is_valid=self.is_valid,
            target_lane_sequence=self.target_lane_sequence,
            tactical_search_space=self.tactical_search_space,
            target_route=new_tr[..., :2],
            route_commands=self.route_commands,
            navigation_goal=self.navigation_goal,
        )

    ### TODO: Change to 3d Transform
    def transform_to_2d(self, matrix, target_ri):
        """
        Transforms the target_route attribute using the provided matrix and assigns it to the target_ri object.

        Args:
            matrix: The transformation matrix to apply to the target_route.
            target_ri: The RoutingInformation object whose target_route will be updated.

        Returns:
            None

        Notes:
            - ADD 3D Transform support if needed.
        """
        if self.target_route is not None:
            target_ri.target_route = transform_2d_seq(self.target_route, matrix)


class RoutingUtils:
    """Utility class for routing and lane sequence operations in map-based environments.
    This class provides a collection of static methods for route generation, lane sequence extraction,
    distance calculations, and pose-to-lane association. It is designed to work with map and trajectory
    representations, supporting tasks such as converting lane sequences to routes, finding the best
    matching lane for a given pose, and generating routing information from waypoints or ego trajectories.

    Methods:
        lanes_sequence_to_route(emap, seq):
            Converts a sequence of lane indices to a route by concatenating their centerlines.
        route_to_sequence(pi, route):
            Converts a route (list of poses) to a sequence of lane indices.
        generate_route_info_from_waypoints(waypoints):
            Generates routing information from a list of waypoints.
        distance_to_line(point, line_start, line_end, epsilon=0.0):
            Computes the distance from a point to a line segment and checks if the projection is within the segment.
        distance_to_line_np(p, l1, l2, epsilon=0.0):
            NumPy version of distance_to_line for vectorized operations.
        get_distance_to_linestring(segment, ls):
            Computes the average distance from a segment's centerline to a linestring.
        get_best_next(pmap, segment, route):
            Finds the best successor segment based on geometric distance to the route.
        get_best_next_slow(pmap, segment, route):
            Brute-force version of get_best_next using overlap count and distance.
        get_route_for_segment(pmap, seg_id, traj):
            Generates a lane sequence for a segment that best matches a given trajectory.
        check_pose_to_lane(lane, p, dist_threshold=8.0, angle_threshold=0.5):
            Checks if a pose is close and aligned to a lane segment.
        check_pose_to_lane_idx(lane, p, dist_threshold=8.0, angle_threshold=0.45, epsilon=0.2):
            Returns the index of the lane segment if the pose is close and aligned, otherwise -1.
        get_lanes_for_pose_dir(pmap, pose, dist_threshold=8.0):
            Returns indices of lanes that match a pose's position and direction.
        get_lanes_for_pose(pmap, pose):
            Returns indices of lanes that contain the given pose.
        get_lanes_for_pose_dir_opt(pmap, p, dist_threshold=5.0):
            Optimized version of get_lanes_for_pose_dir, returns tuples of (lane index, segment index).
        get_distance_along_opt2(centerline, seg_idx, traj, traj_idx, epsilon=0.2, max_miss_count=3):
            Computes the distance along a centerline that matches a trajectory.
        get_route_for_traj_opt2(pmap, lidp, traj, epsilon=0.2, max_miss_count=3):
            Generates a lane sequence for a trajectory using optimized matching.
        generate_route_from_ego_traj(pi):
            Generates a route and lane sequence from the ego vehicle's trajectory.
        generate_route_from_ego_traj_old(pi):
            Legacy version of generate_route_from_ego_traj using brute-force matching.
        generate_target_lane_sequence_from_route(pi):
            Generates and sets the target lane sequence from the current route.
        generate_target_lane_sequence_from_route_map(pmap, route):
            Generates the best lane sequence for a given route using the map.
        generate_target_lane_sequence_from_ego_trajectory(cc):
            Generates the target lane sequence from the ego vehicle's trajectory.
        create_route_info(route):
            Creates a RoutingInformation object from a route with waypoints and commands.
        find_closest_waypoint(waypoints, ego_x, ego_y):
            Finds the index of the closest waypoint to the ego vehicle.
        compute_distance_to_line(waypoints, closest_idx, ego_x, ego_y):
            Computes the distance from the ego vehicle to the line defined by two waypoints.
        compute_relative_yaw(waypoints, closest_idx, ego_yaw):
            Computes the relative yaw between the ego vehicle and the route segment.

    Note:
        This class assumes the existence of supporting classes such as EpisodeMap, RoutingInformation,
        and map/trajectory representations. Some methods are marked as TODO or legacy and may require
        further refactoring or optimization.
    """

    ### TODO: Refactor, this is a very old version!!!
    @staticmethod
    def lanes_sequence_to_route(emap: EpisodeMap, seq):
        """
        Converts a sequence of lane indices into a concatenated route of centerlines.

        Args:
            emap (EpisodeMap): The map object providing access to lane centerlines.
            seq (iterable): Sequence of lane indices.

        Returns:
            np.ndarray: Concatenated array of centerline points representing the route.
        """
        route = []
        for lid in seq:
            centerline = emap.get_centerline_by_idx(lid)
            route.append(centerline)
        troute = np.concatenate(route)
        return troute  # np.unique(troute, axis=0)

    @staticmethod
    def route_to_sequence(pi, route):
        """
        Converts a route of poses into a sequence of unique lane IDs.

        Args:
            pi: An object with a `pmap` attribute that provides a `get_lane_for_pose` method.
            route (iterable): A sequence of poses to be mapped to lane IDs.

        Returns:
            list: A list of unique lane IDs corresponding to the poses in the route.
        """
        sequence = []
        for p in route:
            lid = pi.pmap.get_lane_for_pose(p)
            if lid not in sequence:
                sequence.append(lid)
        return sequence

    ### TODO: implement Interpolation, Lane Sequence and TSS generation
    @staticmethod
    def generate_route_info_from_waypoints(waypoints):
        """
        Generates a RoutingInformation object from a list of waypoints.

        Args:
            waypoints (list): A list of waypoints, where each waypoint is an iterable with at least two elements representing coordinates.

        Returns:
            RoutingInformation: An instance of RoutingInformation with the target route set to the provided waypoints.
        """

        ri = RoutingInformation()
        wps = np.array([p[:2] for p in waypoints])
        ri.target_route = wps

        return ri

    @staticmethod
    def distance_to_line(point, line_start, line_end, epsilon=0.0):
        """
        Calculates the shortest distance from a point to a line segment and checks if the projection falls within the segment.

        Args:
            point (tuple): The (x, y) coordinates of the point.
            line_start (tuple): The (x, y) coordinates of the start of the line segment.
            line_end (tuple): The (x, y) coordinates of the end of the line segment.
            epsilon (float, optional): Tolerance for boundary inclusion. Defaults to 0.0.

        Returns:
            tuple: A tuple containing:
                - distance (float): The shortest distance from the point to the line segment.
                - is_within_segment (bool): True if the projection of the point falls within the segment (with epsilon tolerance), False otherwise.
        """

        px = point[0]
        py = point[1]
        x1 = line_start[0]
        y1 = line_start[1]
        x2 = line_end[0]
        y2 = line_end[1]

        # Calculate the length of the line segment
        line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate the dot product of the vectors
        dot_product = ((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))

        if math.fabs(line_length) < 0.0001:
            return 0, False

        # Calculate the projection of the point onto the line
        projection = dot_product / line_length**2
        # projection = dot_product / line_length

        # Calculate the coordinates of the projected point
        x_proj = x1 + projection * (x2 - x1)
        y_proj = y1 + projection * (y2 - y1)

        # Calculate the distance between the point and the projected point
        distance = math.sqrt((px - x_proj) ** 2 + (py - y_proj) ** 2)

        return distance, -epsilon <= dot_product < line_length + epsilon

    @staticmethod
    def distance_to_line_np(p, l1, l2, epsilon=0.0):
        """
        Calculates the perpendicular distance from a point to a line segment in N-dimensional space using NumPy.

        Args:
            p (np.ndarray): The point from which the distance is measured.
            l1 (np.ndarray): The starting point of the line segment.
            l2 (np.ndarray): The ending point of the line segment.
            epsilon (float, optional): Tolerance for inclusion at the segment's endpoints. Defaults to 0.0.

        Returns:
            tuple:
                float: The perpendicular distance from the point to the line segment.
                bool: True if the projection of the point onto the line lies within the segment (with epsilon tolerance), False otherwise.
        """

        diff = l2 - l1
        line_length_sqrd = np.dot(diff, diff)
        line_length = np.sqrt(line_length_sqrd)
        if np.abs(line_length) < 0.0001:
            return 0, False
        dot_product = np.dot((p - l1), diff)
        projection = l1 + (dot_product / line_length_sqrd) * diff
        distance = np.linalg.norm(p - projection)
        return distance, -epsilon <= dot_product < line_length + epsilon

    @staticmethod
    def get_distance_to_linestring(segment, ls):
        """
        Calculates the average distance from the points in a segment's centerline to a given linestring.

        Args:
            segment: An object with a `centerline` attribute, which is an iterable of points.
            ls: A list of points representing the linestring.

        Returns:
            Tuple[float, int]: A tuple containing the average distance (float) from the segment's centerline points
            to the linestring (considering only those points that project onto a segment of the linestring), and the
            count (int) of such points. If no points project onto the linestring, returns (9999, 0).
        """

        dsum = 0
        dcount = 0
        for p in segment.centerline:
            for i in range(len(ls) - 1):
                l1 = ls[i]
                l2 = ls[i + 1]
                dist, is_in = RoutingUtils.distance_to_line(p, l1, l2)
                if is_in:
                    dsum += dist
                    dcount += 1
        if dcount == 0:
            return 9999, dcount

        return dsum / dcount, dcount

    @staticmethod
    def get_best_next(pmap, segment, route):
        """
        Finds the best next segment from the current segment based on the minimum distance to the given route.

        Args:
            pmap: The map object containing lane and segment information.
            segment: The current segment object with a list of successor indices.
            route: A numpy array representing the route as a sequence of 2D points.

        Returns:
            Tuple[int, float]: The index of the best next segment and the corresponding minimum distance.
        """
        res_id = -1
        res_dist = 999999
        for suc_idx in segment.successors:
            if suc_idx < len(pmap.lanes):
                suc = pmap.get_segment(suc_idx)
                l1 = shapely.geometry.LineString(suc.centerline)
                l2 = shapely.geometry.LineString(route[:, :2])
                dist = l1.distance(l2)

                if dist < res_dist:
                    res_id = suc_idx
                    res_dist = dist
        return res_id, res_dist

    @staticmethod
    def get_best_next_slow(pmap, segment, route):
        """
        Finds the best next segment from the current segment based on distance and overlap with the route.

        Args:
            pmap: The map object containing lane and segment information.
            segment: The current segment object.
            route: The target route linestring.

        Returns:
            Tuple[int, float]: The index of the best next segment and its distance to the route.
        """
        res_id = -1
        res_dist = 999999
        for suc_idx in segment.successors:
            if suc_idx < len(pmap.lanes):
                suc = pmap.get_segment(suc_idx)
                dist, overlap_count = RoutingUtils.get_distance_to_linestring(suc, route)

                if overlap_count > 0 and dist < res_dist:
                    res_id = suc_idx
                    res_dist = dist
        return res_id, res_dist

    @staticmethod
    def get_route_for_segment(pmap, seg_id, traj):
        """
        Computes the route for a given segment based on a trajectory.

        Args:
            pmap: The map object containing segment information and methods.
            seg_id (int): The ID of the starting segment.
            traj: The trajectory information used to determine the best next segment.

        Returns:
            tuple: A tuple containing:
                - seq (list of int): The sequence of segment IDs representing the route.
                - float: The average distance per segment along the route.
        """

        seg = pmap.get_segment(seg_id)

        seq_dist = 0
        seq = []
        seq.append(seg.id)
        while len(seg.successors) > 0:
            next_id, res_dist = RoutingUtils.get_best_next_slow(pmap, seg, traj)
            if next_id in seq:
                break
            if next_id < 0:
                break

            seq_dist += res_dist
            seq.append(next_id)
            seg = pmap.get_segment(next_id)

        return seq, seq_dist / len(seq)

    @staticmethod
    def check_pose_to_lane(lane, p, dist_threshold=8.0, angle_threshold=0.5):
        """
        Checks if a given pose is close to and aligned with a lane centerline segment.

        Args:
            lane: An object with a `centerline` attribute, which is an array-like sequence of points.
            p (array-like): The pose as [x, y, theta], where theta is the orientation in radians.
            dist_threshold (float, optional): Maximum allowed distance from the lane centerline. Defaults to 8.0.
            angle_threshold (float, optional): Maximum allowed angular difference (in cosine similarity). Defaults to 0.5.

        Returns:
            bool: True if the pose is within the distance and angle thresholds of any lane segment, False otherwise.
        """

        pdir = np.array([np.cos(p[2]), np.sin(p[2])])
        for i in range(len(lane.centerline) - 1):
            l1 = lane.centerline[i]
            l2 = lane.centerline[i + 1]

            dist, is_in = RoutingUtils.distance_to_line(p, l1, l2, 0.05)

            if is_in and dist < dist_threshold:
                diff = l2 - l1
                ldir = diff / np.linalg.norm(diff)

                dir_diff = np.dot(pdir, ldir)
                if dir_diff > 1.0 - angle_threshold:
                    return True

        return False

    @staticmethod
    def check_pose_to_lane_idx(lane, p, dist_threshold=8.0, angle_threshold=0.45, epsilon=0.2):
        """
        Checks if a given pose is close to and aligned with any segment of a lane's centerline.

        Args:
            lane: An object with a `centerline` attribute, which is an array-like sequence of points.
            p (array-like): The pose as [x, y, heading], where heading is in radians.
            dist_threshold (float, optional): Maximum allowed distance from the centerline segment. Defaults to 8.0.
            angle_threshold (float, optional): Maximum allowed angular difference (in cosine space) between pose direction and lane segment. Defaults to 0.45.
            epsilon (float, optional): Tolerance for determining if the projection is within the segment. Defaults to 0.2.

        Returns:
            int: The index of the centerline segment that matches the pose, or -1 if none found.
        """

        pdir = np.array([np.cos(p[2]), np.sin(p[2])])

        for i in range(len(lane.centerline) - 1):
            l1 = lane.centerline[i]
            l2 = lane.centerline[i + 1]

            dist, is_in = RoutingUtils.distance_to_line(p, l1, l2, epsilon)

            if is_in and dist < dist_threshold:
                diff = l2 - l1
                ldir = diff / np.linalg.norm(diff)

                dir_diff = np.dot(pdir, ldir)
                if dir_diff > 1.0 - angle_threshold:
                    return i

        return -1

    @staticmethod
    def get_lanes_for_pose_dir(pmap, pose, dist_threshold=8.0):
        """
        Returns a list of lane indices from the map that are within a specified distance threshold of a given pose and satisfy pose-to-lane checks.

        Args:
            pmap: An object containing lane information, expected to have a 'lanes' attribute.
            pose (array-like): The pose to check, where the first two elements represent the position.
            dist_threshold (float, optional): Maximum distance from the lane to consider. Defaults to 8.0.

        Returns:
            list: Indices of lanes that are within the distance threshold and pass the pose-to-lane check.
        """

        res = []
        for i, lane in enumerate(pmap.lanes):
            if lane.distance_to_lane(pose[:2]) > dist_threshold:
                continue
            if RoutingUtils.check_pose_to_lane(lane, pose, dist_threshold):
                res.append(i)

        return res

    @staticmethod
    def get_lanes_for_pose(pmap, pose):
        """
        Returns the indices of lanes in the map that contain the given pose.

        Args:
            pmap: An object representing the map, expected to have a 'lanes' attribute.
            pose: A sequence (e.g., list or numpy array) representing the pose, where the first two elements are the x and y coordinates.

        Returns:
            list: A list of indices of lanes that contain the given pose.
        """

        res = []
        for i, lane in enumerate(pmap.lanes):
            if lane.check_point_is_on(pose[:2]):
                res.append(i)

        return res

    @staticmethod
    def get_lanes_for_pose_dir_opt(pmap, p, dist_threshold=5.0):
        """
        Finds lanes in the map that are within a specified distance threshold of a given pose and returns their indices.

        Args:
            pmap: The map object containing lane information. Must have a 'lanes' attribute.
            p (array-like): The pose as a sequence (e.g., [x, y, ...]).
            dist_threshold (float, optional): The maximum distance to consider a lane as nearby. Defaults to 5.0.

        Returns:
            list of tuple: A list of tuples, each containing the lane index and the corresponding pose-to-lane index.
        """

        res = []
        for i, lane in enumerate(pmap.lanes):
            if not lane.is_in_circle(p[:2], dist_threshold):
                continue
            if lane.distance_to_lane(p[:2]) > dist_threshold:
                continue

            idx = RoutingUtils.check_pose_to_lane_idx(lane, p, dist_threshold=dist_threshold)
            if idx >= 0:
                res.append((i, idx))
        return res

    @staticmethod
    def get_distance_along_opt2(centerline, seg_idx, traj, traj_idx, epsilon=0.2, max_miss_count=3):
        """
        Calculates the cumulative distance along a centerline that a trajectory follows, starting from given indices.

        Args:
            centerline (list): List of points representing the centerline.
            seg_idx (int): Starting index on the centerline.
            traj (list): List of trajectory points.
            traj_idx (int): Starting index in the trajectory.
            epsilon (float, optional): Tolerance for considering a point as being on the segment. Defaults to 0.2.
            max_miss_count (int, optional): Maximum number of centerline segments to check for each trajectory point. Defaults to 3.

        Returns:
            tuple: A tuple containing:
                - res_dist (float): Total distance accumulated along the centerline.
                - res_count (int): Number of trajectory points matched to the centerline.
                - res_traj_idx (int): Index of the last matched trajectory point.
        """

        res_dist = 0
        res_count = 0
        res_traj_idx = 0
        for i in range(traj_idx, len(traj)):
            p = traj[i]
            found = False
            for s in range(seg_idx, min(seg_idx + max_miss_count, len(centerline)) - 1):
                w1 = centerline[s]
                w2 = centerline[s + 1]
                dist, is_in = RoutingUtils.distance_to_line(p, w1, w2, epsilon=epsilon)

                if is_in:
                    res_dist += dist
                    res_count += 1
                    found = True
                    seg_idx = s
                    res_traj_idx = i
                    break
            if not found:
                break

        return res_dist, res_count, res_traj_idx

    @staticmethod
    def get_route_for_traj_opt2(pmap, lidp, traj, epsilon=0.2, max_miss_count=3):
        """
        Computes the optimal route through a map based on a given trajectory, starting from a specified segment.

        Args:
            pmap: The map object containing segments and their relationships.
            lidp (tuple): A tuple (lid, seg_idx) specifying the starting segment ID and index.
            traj (list): The trajectory to match against the map segments.
            epsilon (float, optional): Tolerance for matching trajectory points. Defaults to 0.2.
            max_miss_count (int, optional): Maximum allowed consecutive misses when matching. Defaults to 3.

        Returns:
            tuple: A tuple (seq, avg_dist) where:
                seq (list): List of segment IDs representing the computed route.
                avg_dist (float): Average distance per matched segment along the route.
        """

        lid = lidp[0]
        seg_idx = lidp[1]
        res_dist = 0
        res_count = 0
        segment = pmap.get_segment(lid)
        seq = []
        tdist, tcount, traj_idx = RoutingUtils.get_distance_along_opt2(
            segment.centerline,
            seg_idx,
            traj,
            0,
            epsilon=epsilon,
            max_miss_count=max_miss_count,
        )
        seq.append(lid)
        res_dist += tdist
        res_count += tcount
        while len(segment.successors) > 0:
            best_succ = -1
            best_dist = 99999999
            best_traj_idx = -1
            for sidx in segment.successors:
                succ = pmap.get_segment(sidx)
                tdist, tcount, ttraj_idx = RoutingUtils.get_distance_along_opt2(
                    succ.centerline,
                    0,
                    traj,
                    traj_idx,
                    epsilon=epsilon,
                    max_miss_count=max_miss_count,
                )

                if tdist < best_dist:
                    best_dist = tdist
                    best_succ = sidx
                    best_traj_idx = ttraj_idx

            res_dist += tdist
            res_count += tcount
            traj_idx = best_traj_idx
            segment = pmap.get_segment(best_succ)
            seq.append(best_succ)

        return seq, res_dist / res_count

    @staticmethod
    def generate_route_from_ego_traj(pi):
        """
        Generates a route for the ego vehicle based on its trajectory.
        This function computes the best lane sequence and route for the ego vehicle
        using its historical and future trajectory, and updates the routing information
        in the provided planning interface.

        Args:
            pi: Planning interface object containing the ego vehicle, map, and methods
                for setting routing information.

        Returns:
            bool: True if a valid route was generated and set, False otherwise.
        """

        ego = pi.get_ego()
        lids = RoutingUtils.get_lanes_for_pose_dir_opt(pi.pmap, ego.hist[-1], dist_threshold=7.0)

        if len(lids) == 0:
            return False
        best_dis = 9999999999
        best_seq = None

        traj = np.concatenate((ego.hist[-1].reshape(1, -1), ego.future), axis=0)
        for lidp in lids:
            seq, seq_dist = RoutingUtils.get_route_for_traj_opt2(pi.pmap, lidp, traj)
            if seq_dist < best_dis:
                best_seq = seq
                best_dis = seq_dist

        successors = pi.pmap.get_segment(best_seq[-1]).successors
        while len(successors) > 0:
            next_succ = successors[0]
            best_seq.append(next_succ)
            successors = pi.pmap.get_segment(next_succ).successors

        route = RoutingUtils.lanes_sequence_to_route(pi, best_seq)

        ri = RoutingInformation()

        ri.target_lane_sequence = best_seq
        ri.tactical_search_space = best_seq
        ri.target_route = route

        pi.set_routing_info(ri)

        return True

    ## Old, slower version. Kept as brute force but accurate solution. Compare new version to this one.
    @staticmethod
    def generate_route_from_ego_traj_old(pi):
        """
        Generates a route for the ego vehicle based on its historical and future trajectory.
        This function determines the best lane sequence and route for the ego vehicle by analyzing its trajectory,
        selecting the optimal lane, and extending the route through successor segments. The resulting routing
        information is set in the provided planning interface.

        Args:
            pi: The planning interface object containing the ego vehicle, map, and routing utilities.

        Returns:
            bool: True if a valid route was generated and set, False otherwise.

        Notes:
            ## Old, slower version. Kept as brute force but accurate solution. Compare new version to this one.

        """

        ego = pi.get_ego()
        traj = np.concatenate((ego.hist, ego.future), axis=0)

        lids = RoutingUtils.get_lanes_for_pose_dir(pi.pmap, ego.hist[-1])

        if len(lids) == 0:
            return False
        best_dis = 9999999999
        best_seq = None

        for lid in lids:
            seq, seq_dist = RoutingUtils.get_route_for_segment(pi.pmap, lid, traj)
            if seq_dist < best_dis:
                best_seq = seq
                best_dis = seq_dist

        successors = pi.pmap.get_segment(best_seq[-1]).successors
        while len(successors) > 0:
            next_succ = successors[0]
            best_seq.append(next_succ)
            successors = pi.pmap.get_segment(next_succ).successors

        route = RoutingUtils.lanes_sequence_to_route(pi, best_seq)

        ri = RoutingInformation()

        ri.target_lane_sequence = best_seq
        ri.tactical_search_space = best_seq
        ri.target_route = route

        pi.set_routing_info(ri)

        return True

    @staticmethod
    def generate_target_lane_sequence_from_route(pi):
        """
        Generates and sets the target lane sequence for the current route.
        This function computes the best target lane sequence based on the provided
        route and map, and updates the routing information with the resulting
        sequence and tactical search space.

        Args:
            pi: An object containing routing information and the map (pmap).

        Returns:
            None

        Notes:
            TODO: Check if croping to nearest ego pos is required.
        """

        ## TODO: Check if croping to nearest ego pos is required.
        target_route = pi.routing_information.target_route  # [closest_index:]

        best_seq = RoutingUtils.generate_target_lane_sequence_from_route_map(pi.pmap, target_route)

        pi.routing_information.set_target_lane_sequence(best_seq)
        pi.routing_information.set_tactical_search_space(best_seq)

    @staticmethod
    def generate_target_lane_sequence_from_route_map(pmap, route):
        """
        Generates the optimal sequence of lane IDs from a route map given a trajectory.

        Args:
            pmap: The map object providing lane and segment information.
            route: A list of poses representing the trajectory to follow.

        Returns:
            list: The sequence of lane IDs that best matches the given trajectory.
        """

        traj = route

        lids = pmap.get_lanes_for_pose(traj[0])

        best_dis = 9999999999
        best_seq = None

        for lid in lids:
            seq, seq_dist = RoutingUtils.get_route_for_segment(pmap, lid, traj)
            if seq_dist < best_dis:
                best_seq = seq
                best_dis = seq_dist

        # successors = pi.pmap.get_segment(best_seq[-1]).successors
        # while len(successors) > 0:
        #     next_succ = successors[0]
        #     if next_succ in best_seq:
        #         break
        #     best_seq.append(next_succ)
        #     successors = pi.pmap.get_segment(next_succ).successors

        print(best_seq)
        return best_seq

    @staticmethod
    def generate_target_lane_sequence_from_ego_trajectory(
        cc: "CoreContainer",
        t0: int = 0,
        xy_threshold: float = 5.0,
        angle_threshold: float = 0.7,
        successor_distance_threshold: float = 8.0,
        max_n_segments_per_route: int = 30,
        max_n_routes_per_starting_lane: int = 10,
        verbose: bool = False,
    ) -> list[np.ndarray]:
        """
        Generate target lane sequence from ego trajectory by finding lanes that best match the vehicle's path.

        Algorithm:
        1. Create linestrings for all lane centerlines and ego trajectory
        2. Find candidate lanes within distance and angle thresholds to ego pose at t0
        3. Look up corresponding lane_infos for candidate lanes
        4. Generate all possible routes using successors and neighbors
        5. Score each route using average distance to ego trajectory and return the best one

        Args:
            cc ("CoreContainer"): Container with episode data including map and odometry
            t0 (int): Starting time step for trajectory analysis. Defaults to 0.
            xy_threshold (float): Distance threshold for initial lane matching. Defaults to 5.0.
            angle_threshold (float): Cosine similarity threshold for angle matching (smaller = stricter). Defaults to 0.7.
            successor_distance_threshold (float): Distance threshold for adding successors/neighbors. Defaults to 8.0.
            max_n_segments_per_route (int): Maximum number of lane segments in a route. Defaults to 30.
            max_n_routes_per_starting_lane (int): Limit routes generated per starting lane. Defaults to 10.
            verbose (bool): Enable verbose output. Defaults to False.

        Returns:
            list[np.ndarray]: List of lane centerline arrays representing the best route sequence
        """
        assert cc.odometry is not None and cc.odometry.transform is not None
        assert cc.map_container is not None

        if verbose:
            print("Starting target lane sequence generation")

        # Each one is an array of shape [t]
        x, y, yaw = TransformsOperations.get_xyyaw_from_transforms(cc.odometry.transform[0, :])
        # Combine everything into one ego state of shape [t, 3]
        ego_state = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), yaw.reshape(-1, 1)), axis=1)

        # Extract ego trajectory from t0 onwards
        ego_trajectory = ego_state[t0:, :2]  # Only x, y coordinates for trajectory
        if len(ego_trajectory) < 2:
            if verbose:
                print("Ego trajectory too short for route generation")
            return []

        # Step 1: Create linestrings for all lane centerlines and ego trajectory
        if verbose:
            print("Creating linestrings for lane centerlines")
        episode_map = cc.map_container.maps[0]
        lane_linestrings = {}

        for i, lane_info in enumerate(episode_map.lane_infos):
            # Get centerline for this lane
            centerline_start = lane_info.centerline_indices[0]
            centerline_end = lane_info.centerline_indices[1]
            centerline_points = episode_map.lane_centerlines[centerline_start:centerline_end, :2]  # Only x, y

            if len(centerline_points) >= 2:
                lane_linestrings[lane_info.lane_id] = shapely.geometry.LineString(centerline_points)

        if verbose:
            print("Creating ego trajectory linestring")
        ego_linestring = shapely.geometry.LineString(ego_trajectory)

        # Step 2: Find candidate lanes within distance and angle thresholds to ego pose at t0
        if verbose:
            print("Finding candidate lanes within thresholds to ego pose at t0")
        candidate_lane_ids = []

        # Get ego state at t0 for position and direction comparison
        ego_pose_t0 = ego_state[t0]  # [x, y, yaw]
        ego_point = shapely.geometry.Point(ego_pose_t0[0], ego_pose_t0[1])
        ego_direction = np.array([np.cos(ego_pose_t0[2]), np.sin(ego_pose_t0[2])])

        for lane_id, lane_linestring in lane_linestrings.items():
            # Check distance from ego pose at t0 to lane centerline
            distance_to_lane = lane_linestring.distance(ego_point)

            if distance_to_lane <= xy_threshold:
                # Check angle alignment - get lane direction at closest point
                lane_info_idx = episode_map.lane_id_to_idx[lane_id]
                lane_info = episode_map.lane_infos[lane_info_idx]

                centerline_start = lane_info.centerline_indices[0]
                centerline_end = lane_info.centerline_indices[1]
                centerline_points = episode_map.lane_centerlines[centerline_start:centerline_end, :2]

                # Find direction of first segment of the lane
                if len(centerline_points) >= 2:
                    # Find the direction of the lane at the closest point
                    # Get a small distance along the line to calculate direction
                    distance_along_line = lane_linestring.project(ego_point)
                    line_length = lane_linestring.length

                    # Calculate direction by looking at a small segment around the closest point
                    delta = min(1.0, line_length * 0.01)  # 1m or 1% of line length, whichever is smaller

                    # Get points slightly before and after the closest point
                    point_before = lane_linestring.interpolate(max(0, distance_along_line - delta))
                    point_after = lane_linestring.interpolate(min(line_length, distance_along_line + delta))

                    # Calculate direction vector from before to after
                    lane_dir_at_closest = np.array([point_after.x - point_before.x, point_after.y - point_before.y])

                    # Normalize the direction vector
                    if np.linalg.norm(lane_dir_at_closest) > 0:
                        lane_dir_at_closest = lane_dir_at_closest / np.linalg.norm(lane_dir_at_closest)

                        # Check angle similarity
                        dot_product = np.dot(ego_direction, lane_dir_at_closest)
                        if dot_product > angle_threshold:  # Cosine similarity threshold
                            candidate_lane_ids.append(lane_id)

        if verbose:
            print(f"Found {len(candidate_lane_ids)} candidate lanes")

        # Step 3: Look up corresponding lane_infos for candidate lanes
        candidate_lane_infos = []
        for lane_id in candidate_lane_ids:
            lane_info_idx = episode_map.lane_id_to_idx[lane_id]
            candidate_lane_infos.append((lane_id, episode_map.lane_infos[lane_info_idx]))

        # Step 4: Generate all possible routes using successors and neighbors
        if verbose:
            print("Generating possible routes from candidate lanes")
        all_routes = []

        def generate_route_from_lane(start_lane_id, visited_lanes=None, current_depth=0):
            """Recursively generate routes from a starting lane with depth limiting"""
            if visited_lanes is None:
                visited_lanes = set()

            if current_depth >= max_n_segments_per_route:
                return [[start_lane_id]]

            if start_lane_id in visited_lanes:
                return [[start_lane_id]]

            visited_lanes_copy = visited_lanes.copy()
            visited_lanes_copy.add(start_lane_id)

            # Get current lane info
            if start_lane_id not in episode_map.lane_id_to_idx:
                return [[start_lane_id]]

            lane_info_idx = episode_map.lane_id_to_idx[start_lane_id]
            lane_info = episode_map.lane_infos[lane_info_idx]

            # Get all potential next lanes (successors and neighbors)
            next_lane_ids = []
            next_lane_ids.extend(lane_info.successors)
            # NOTE This is only tested using successors. Lane changes may introduce errors if not handled properly.
            next_lane_ids.extend(lane_info.lane_change_to_left)
            next_lane_ids.extend(lane_info.lane_change_to_right)

            # Filter next lanes by distance to ego trajectory
            valid_next_lanes = []
            for next_lane_id in next_lane_ids:
                if next_lane_id not in episode_map.lane_id_to_idx:
                    continue

                if next_lane_id in visited_lanes_copy:
                    continue

                if next_lane_id in lane_linestrings:
                    next_lane_linestring = lane_linestrings[next_lane_id]
                    distance_to_traj = next_lane_linestring.distance(ego_linestring)

                    if distance_to_traj <= successor_distance_threshold:
                        # Also check angle alignment with ego trajectory
                        # Use shortest_line to find the actual closest points between the linestrings
                        try:
                            shortest_line = shapely.shortest_line(ego_linestring, next_lane_linestring)
                            closest_point_on_ego = shortest_line.coords[0]  # First point is on ego trajectory
                            closest_point_on_lane = shortest_line.coords[1]  # Second point is on successor lane

                            # Convert to Points for easier manipulation
                            ego_closest_point = shapely.geometry.Point(closest_point_on_ego)
                            lane_closest_point = shapely.geometry.Point(closest_point_on_lane)

                            # Get ego trajectory direction at the closest point
                            distance_along_ego = ego_linestring.project(ego_closest_point)
                            ego_length = ego_linestring.length

                            # Get successor lane direction at its closest point
                            distance_along_lane = next_lane_linestring.project(lane_closest_point)
                            lane_length = next_lane_linestring.length

                            if ego_length > 0 and lane_length > 0:
                                # Calculate ego direction at closest point
                                ego_delta = min(1.0, ego_length * 0.01)  # 1m or 1% of trajectory length
                                ego_point_before = ego_linestring.interpolate(max(0, distance_along_ego - ego_delta))
                                ego_point_after = ego_linestring.interpolate(
                                    min(ego_length, distance_along_ego + ego_delta)
                                )

                                ego_dir_at_closest = np.array(
                                    [ego_point_after.x - ego_point_before.x, ego_point_after.y - ego_point_before.y]
                                )

                                # Calculate successor lane direction at closest point
                                lane_delta = min(1.0, lane_length * 0.01)  # 1m or 1% of lane length
                                lane_point_before = next_lane_linestring.interpolate(
                                    max(0, distance_along_lane - lane_delta)
                                )
                                lane_point_after = next_lane_linestring.interpolate(
                                    min(lane_length, distance_along_lane + lane_delta)
                                )

                                lane_dir_at_closest = np.array(
                                    [lane_point_after.x - lane_point_before.x, lane_point_after.y - lane_point_before.y]
                                )

                                # Normalize both direction vectors
                                if np.linalg.norm(ego_dir_at_closest) > 0 and np.linalg.norm(lane_dir_at_closest) > 0:
                                    ego_dir_normalized = ego_dir_at_closest / np.linalg.norm(ego_dir_at_closest)
                                    lane_dir_normalized = lane_dir_at_closest / np.linalg.norm(lane_dir_at_closest)

                                    # Check angle similarity
                                    dot_product = np.dot(ego_dir_normalized, lane_dir_normalized)
                                    if dot_product > angle_threshold:  # Both distance and angle criteria met
                                        valid_next_lanes.append(next_lane_id)
                                else:
                                    # If we can't compute direction, fall back to distance-only check
                                    valid_next_lanes.append(next_lane_id)
                            else:
                                # If linestrings are too short, fall back to distance-only check
                                valid_next_lanes.append(next_lane_id)

                        except Exception as e:
                            # If shortest_line fails for any reason, fall back to distance-only check
                            valid_next_lanes.append(next_lane_id)
                            print(f"Exception during shortest_line calculation: {str(e)}")

            if not valid_next_lanes:
                # No valid successors, end route here
                return [[start_lane_id]]

            # Generate routes for each valid next lane (limit to prevent explosion)
            routes = []
            for i, next_lane_id in enumerate(valid_next_lanes):
                if i >= max_n_routes_per_starting_lane:  # Limit routes per lane
                    break

                sub_routes = generate_route_from_lane(next_lane_id, visited_lanes_copy, current_depth + 1)
                for sub_route in sub_routes[:3]:  # Limit sub-routes
                    routes.append([start_lane_id] + sub_route)

            if not routes:
                routes = [[start_lane_id]]

            return routes

        # Generate routes from each candidate starting lane
        for lane_id, _ in candidate_lane_infos:
            routes_from_lane = generate_route_from_lane(lane_id)
            all_routes.extend(routes_from_lane)

            # Limit total number of routes to prevent memory issues
            if len(all_routes) > 1000:
                break

        if verbose:
            print(f"Generated {len(all_routes)} possible routes")

        # Step 5: Score each route using average distance to ego trajectory
        if verbose:
            print("Scoring routes based on distance to ego trajectory")
        scored_routes = []

        for route in all_routes:
            total_distance = 0.0
            total_segments = 0

            for lane_id in route:
                if lane_id in lane_linestrings:
                    lane_linestring = lane_linestrings[lane_id]
                    distance = lane_linestring.distance(ego_linestring)
                    total_distance += distance
                    total_segments += 1

            if total_segments > 0:
                average_distance = total_distance / total_segments
                scored_routes.append((route, average_distance))

        # Sort routes by score (lower distance is better)
        scored_routes.sort(key=lambda x: x[1])

        if scored_routes:
            if verbose:
                print(f"Best route has average distance: {scored_routes[0][1]:.3f}m")
        else:
            if verbose:
                print("No valid routes found")
            return []

        # Convert best route to target lane sequence (list of centerline arrays)
        target_lane_sequence = scored_routes[0][0]

        if verbose:
            print(f"Generated target lane sequence with {len(target_lane_sequence)} lane segments")
        return target_lane_sequence

    @staticmethod
    def create_route_info(route):
        """
        Creates a RoutingInformation object from a given route.

        Args:
            route (list): A list of waypoints, where each waypoint is a tuple containing
                a position (with at least two elements) and a route command.

        Returns:
            RoutingInformation: An object containing the target route as a NumPy array
            of positions and a list of route commands.
        """
        res_route = RoutingInformation._create_empty()
        target_route = []
        for wp in route:
            pos = wp[0][:2]
            target_route.append(pos[:2])
            res_route.route_commands.append(wp[1])
        res_route.target_route = np.array(target_route)
        return res_route

    @staticmethod
    def find_closest_waypoint(waypoints, ego_x, ego_y):
        """
        Find the index of the closest waypoint to the ego vehicle.

        Parameters:
            waypoints (np.ndarray): Array of shape (l, 2) containing (x, y) waypoints.
            ego_x (float): X coordinate of the ego vehicle.
            ego_y (float): Y coordinate of the ego vehicle.

        Returns:
            int: Index of the closest waypoint.
        """
        distances = np.linalg.norm(waypoints - np.array([ego_x, ego_y]), axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx

    @staticmethod
    def compute_distance_to_line(waypoints, closest_idx, ego_x, ego_y):
        """
        Compute the distance from the ego vehicle to the line passing through the closest
        waypoint and the next one.

        Parameters:
            waypoints (np.ndarray): Array of shape (l, 2) containing (x, y) waypoints.
            closest_idx (int): Index of the closest waypoint.
            ego_x (float): X coordinate of the ego vehicle.
            ego_y (float): Y coordinate of the ego vehicle.

        Returns:
            float: Distance between the ego and the line.
        """
        if closest_idx == len(waypoints) - 1:
            closest_idx -= 1
        wp1 = waypoints[closest_idx]
        wp2 = waypoints[closest_idx + 1]

        # Compute the line equation: Ax + By + C = 0
        A = wp2[1] - wp1[1]  # y2 - y1
        B = wp1[0] - wp2[0]  # x1 - x2
        C = wp2[0] * wp1[1] - wp1[0] * wp2[1]  # x2*y1 - x1*y2

        denominator = np.sqrt(A**2 + B**2)
        if denominator == 0:
            return 0.0

        distance_to_line = abs(A * ego_x + B * ego_y + C) / denominator
        if np.isnan(distance_to_line):
            return 0.0

        return distance_to_line

    @staticmethod
    def compute_relative_yaw(waypoints, closest_idx, ego_yaw):
        """
        Compute the relative yaw of the ego vehicle with respect to the line passing
        through the closest waypoint and the next one.

        Parameters:
            waypoints (np.ndarray): Array of shape (l, 2) containing (x, y) waypoints.
            closest_idx (int): Index of the closest waypoint.
            ego_yaw (float): Heading angle (yaw) of the ego vehicle in radians.

        Returns:
            float: Relative yaw in radians.
        """
        if closest_idx == len(waypoints) - 1:
            closest_idx -= 1
        wp1 = waypoints[closest_idx]
        wp2 = waypoints[closest_idx + 1]

        # Compute yaw of the line (heading of the segment)
        line_yaw = np.arctan2(wp2[1] - wp1[1], wp2[0] - wp1[0])

        # Compute relative yaw
        relative_yaw = ego_yaw - line_yaw

        # Normalize relative yaw to [-pi, pi]
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
        return relative_yaw
