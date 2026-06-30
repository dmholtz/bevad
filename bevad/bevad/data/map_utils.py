import math
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KDTree


@dataclass(kw_only=True)
class KdTreeMap:
    # lane info
    lane_points: list[np.ndarray]
    lane_types: list[str]

    # trigger volumes info
    trigger_volumes: list[np.ndarray]
    trigger_volume_types: list[str]

    # required for querying
    lane_tree: KDTree
    lane_indices: np.ndarray
    tv_tree: KDTree

    @classmethod
    def build(cls, info_obj: dict):
        # lanes
        lane_sample_points = np.vstack(
            [lsp[:, :2] for lsp in info_obj["lane_sample_points"]]
        )
        lane_indices = []
        for i, lsp in enumerate(info_obj["lane_sample_points"]):
            lane_indices.extend([i] * len(lsp))
        lane_indices = np.array(lane_indices)
        lane_tree = KDTree(lane_sample_points)

        # trigger volumes
        tv_sample_points = np.vstack(
            [tv[:2] for tv in info_obj["trigger_volumes_sample_points"]]
        )
        tv_tree = KDTree(tv_sample_points)

        return cls(
            lane_points=info_obj["lane_points"],
            lane_types=info_obj["lane_types"],
            trigger_volumes=info_obj["trigger_volumes_points"],
            trigger_volume_types=info_obj["trigger_volumes_types"],
            lane_tree=lane_tree,
            lane_indices=lane_indices,
            tv_tree=tv_tree,
        )

    def get_nearby_lanes(
        self,
        point: np.ndarray,
        radius: float,
        crop_to_radius: bool = True,
        x_tf_world: np.ndarray = None,
    ):
        """Return the nearby lanes within a certain radius to the given point.

        Args:
            point (np.ndarray): Shape: (2,). The point to query.
        """

        # query_radius supports batching, but we only query one point at a time
        nearby_indices = self.lane_tree.query_radius(point.reshape(1, 2), r=radius)[0]
        lane_indices = np.unique(self.lane_indices[nearby_indices])

        nearby_lanes = []
        for lane_index in lane_indices:
            lane_points = self.lane_points[lane_index][:, :3]
            lane_type = self.lane_types[lane_index]

            # [optional]: crop the lane segment to the search radius
            if crop_to_radius:
                distances = np.linalg.norm(lane_points[:, :2] - point, axis=1)
                within_radius = distances <= radius
                indices = np.where(within_radius)[0]
                first_idx = indices[0]
                last_idx = indices[-1] + 1
                segment = lane_points[first_idx:last_idx]
            else:
                segment = lane_points

            # [optional]: transform the map from world coordinates to another coordinate system x
            if x_tf_world is not None:
                segment = np.hstack((segment, np.ones_like(segment[:, 0:1])))
                segment = x_tf_world @ segment.T
                segment = segment[:2, :].T  # keep only x and y coordinates

            nearby_lanes.append((segment, lane_type))

        return nearby_lanes

    def get_nearby_trigger_volumes(
        self, point: np.ndarray, radius: float, x_tf_world: np.ndarray = None
    ):
        """Return the nearby trigger volumes within a certain radius to the given point.

        Args:
            point (np.ndarray): Shape: (2,). The point to query.
        """

        # query_radius supports batching, but we only query one point at a time
        nearby_indices = self.tv_tree.query_radius(point.reshape(1, 2), r=radius)[0]
        trigger_volumes = []
        for index in nearby_indices:
            tv_points = self.trigger_volumes[index]
            tv_type = self.trigger_volume_types[index]

            # [optional]: transform the map from world coordinates to another coordinate system x
            if x_tf_world is not None:
                tv_points = np.hstack((tv_points, np.ones_like(tv_points[:, 0:1])))
                tv_points = x_tf_world @ tv_points.T
                tv_points = tv_points[:2, :].T  # keep only x and y coordinates

            trigger_volumes.append((tv_points, tv_type))

        return trigger_volumes


from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Rasterizer:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    H: int
    W: int

    def __post_init__(self):
        self._layers = []

        x_scale = self.W / (self.x_max - self.x_min)
        y_scale = self.H / (self.y_max - self.y_min)
        assert abs(x_scale - y_scale) < 1e-5, (
            f"Image aspect ratio {self.W}/{self.H} does not match patch aspect ratio {(self.x_max - self.x_min)}/{(self.y_max - self.y_min)}"
        )

        self.scale = x_scale

    def _empty_layer(self):
        return np.zeros((self.H, self.W), dtype=np.uint8)

    def add_polyline_layer(self, polylines, width: float):
        # Create a blank image with the specified resolution
        layer = self._empty_layer()

        thickness = max(1, int(width * self.scale))

        for polyline in polylines:
            polyline = polyline[
                :, :2
            ]  # Ensure we only use the first two columns (x, y)
            # Filter points that fall within the specified patch
            filtered_points = polyline[
                (polyline[:, 0] >= self.x_min)
                & (polyline[:, 0] <= self.x_max)
                & (polyline[:, 1] >= self.y_min)
                & (polyline[:, 1] <= self.y_max)
            ]

            if len(filtered_points) > 0:
                # Scale points to fit the image resolution
                scaled_points = np.zeros_like(filtered_points)
                scaled_points[:, 0] = (filtered_points[:, 0] - self.x_min) * self.scale
                scaled_points[:, 1] = (filtered_points[:, 1] - self.y_min) * self.scale

                # Convert points to integer coordinates
                scaled_points = scaled_points.astype(np.int32)

                # Draw the polyline on the image
                cv2.polylines(
                    layer,
                    [scaled_points],
                    isClosed=False,
                    color=255,
                    thickness=thickness,
                )

        self._layers.append(layer)

    def add_polygon_layer(self, polygons):
        # Create a blank image with the specified resolution
        layer = self._empty_layer()

        for polygon in polygons:
            # Filter points that fall within the specified patch
            polygon = polygon[:, :2]
            filtered_points = polygon[
                (polygon[:, 0] >= self.x_min)
                & (polygon[:, 0] <= self.x_max)
                & (polygon[:, 1] >= self.y_min)
                & (polygon[:, 1] <= self.y_max)
            ]

            if len(filtered_points) >= 3:  # Ensure we have a valid polygon
                # Scale points to fit the image resolution
                scaled_points = np.zeros_like(filtered_points)
                scaled_points[:, 0] = (filtered_points[:, 0] - self.x_min) * self.scale
                scaled_points[:, 1] = (filtered_points[:, 1] - self.y_min) * self.scale

                # Convert points to integer coordinates and reshape for OpenCV
                scaled_points = scaled_points.astype(np.int32).reshape((-1, 1, 2))

                # Draw the polygon on the image
                cv2.fillPoly(layer, [scaled_points], color=255)
            else:
                pass
                # print(f"Polygon has {len(filtered_points)} points, expected 4. Skipping.")

        self._layers.append(layer)

    def stack_results(self):
        return np.stack(self._layers, axis=0)


class MapExtractor:
    def __init__(self, map_file):
        self.maps = {
            map_name: KdTreeMap.build(map_info)
            for map_name, map_info in map_file.items()
        }

    def get_map_segmentation(
        self, map_name: str, world_tf_ego, bev_range: float, bev_shape: int
    ):
        # get map
        map = self.maps[map_name]

        # get map elements
        radius = math.sqrt(2 * bev_range**2)
        ego_tf_world = np.linalg.inv(world_tf_ego)
        lanes = map.get_nearby_lanes(
            world_tf_ego[:2, 3],
            radius=radius,
            crop_to_radius=True,
            x_tf_world=ego_tf_world,
        )
        trigger_volumes = map.get_nearby_trigger_volumes(
            world_tf_ego[:2, 3], radius=radius, x_tf_world=ego_tf_world
        )

        lane_config = (
            ("Center", 3),
            ("Broken", 0.1),
            ("Solid", 0.1),
            ("SolidSolid", 0.1),
        )
        polygon_config = ("StopSign",)

        rasterizer = Rasterizer(
            x_min=-bev_range,
            x_max=bev_range,
            y_min=-bev_range,
            y_max=bev_range,
            H=bev_shape[0],
            W=bev_shape[1],
        )

        layers = []

        # lanes
        for lane_type, width in lane_config:
            lines = []
            for line, line_type in lanes:
                if line_type == lane_type:
                    lines.append(line)
            rasterizer.add_polyline_layer(lines, width=width)
            layers.append(lane_type)

        # polygons
        for polygon_type in polygon_config:
            polygons = []
            for tv, tv_type in trigger_volumes:
                if tv_type == polygon_type:
                    polygons.append(tv)
            rasterizer.add_polygon_layer(polygons)
            layers.append(polygon_type)

        # rasterization x-right, y-up coordinates
        rasterization = rasterizer.stack_results()

        # rotate to x-up, y-left coordinates, matching the vehicle coordinate system
        rasterization = np.copy(np.rot90(rasterization, k=-1, axes=(1, 2)))
        return layers, rasterization


"""
import pickle
import time
with open("data/infos/b2d_map_infos.pkl", "rb") as f:
    map_infos = pickle.load(f)

map_extractor = MapExtractor(map_infos)


layers, segmentation = map_extractor.get_map_segmentation("Town10HD", world_tf_ego=np.eye(4), bev_range=80, bev_shape=(400, 400))

segmentation = np.stack(
    (segmentation[0].clip(0,1), (segmentation[1]+segmentation[2]+segmentation[3]).clip(0,1), segmentation[4].clip(0,1)),
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(np.flipud(segmentation.transpose(1, 2, 0).astype(np.float32)))
fig.savefig("map.png")

pass
"""
