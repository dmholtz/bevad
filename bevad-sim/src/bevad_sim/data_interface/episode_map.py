from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from bevad_sim.data_interface.base_entity import BaseEntity

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt


@dataclass
class LaneSegmentInfo:
    """Describes metadata of a lane segment within the road graph.

    Attributes:
        centerline_indices: Index range into the polyline array for the centerline: "lane_centerlines".
        left_boundary_indices: Index range for the left lane boundary polyline: "lane_left_boundaries".
        right_boundary_indices: Index range for the right lane boundary polyline: "lane_right_boundaries".
        lane_type: Lane type enum (e.g., normal, bus).
        left_boundary_type: Boundary type enum for the left side.
        right_boundary_type: Boundary type enum for the right side.
        successors: IDs of succeeding lane segments.
        predecessors: IDs of preceding lane segments.
        lane_change_to_left: IDs of lanes accessible by left lane change.
        lane_change_to_right: IDs of lanes accessible by right lane change.
        lane_id: Unique identifier for this lane segment.
        speed_limit: Speed limit (in m/s or km/h depending on convention).
        is_poly_lane: Whether this lane is a polyline-based lane.
    """

    centerline_indices: tuple[int, int]
    left_boundary_indices: tuple[int, int]
    right_boundary_indices: tuple[int, int]

    lane_type: int
    left_boundary_type: int
    right_boundary_type: int
    successors: list[int]
    predecessors: list[int]
    lane_change_to_left: list[int]
    lane_change_to_right: list[int]

    lane_id: int
    speed_limit: float

    is_poly_lane: bool

    @classmethod
    def create_empty(cls):
        """Creates an empty LaneSegmentInfo with default values."""

        return LaneSegmentInfo(
            centerline_indices=(0, 0),
            left_boundary_indices=(0, 0),
            right_boundary_indices=(0, 0),
            lane_type=-1,
            left_boundary_type=-1,
            right_boundary_type=-1,
            successors=[],
            predecessors=[],
            lane_change_to_left=[],
            lane_change_to_right=[],
            lane_id=-1,
            speed_limit=-1.0,
            is_poly_lane=False,
        )


@dataclass
class TrafficControlElementInfo:
    """Describes a traffic control element such as a light or sign.

    Attributes:
        geometry_box_indices: Index range into geometry representation: "tce_geometry_boxes".
        control_poly_indices: Index range for control polygon: "tce_control_polys".
        traffic_control_element_type: Enum type of the traffic control element.
        traffic_control_element_id: Unique identifier.
    """

    geometry_box_indices: tuple[int, int]
    control_poly_indices: tuple[int, int]

    traffic_control_element_type: int
    traffic_control_element_id: int

    @classmethod
    def create_empty(cls):
        """Creates an empty TrafficControlElementInfo with default values."""

        return TrafficControlElementInfo(
            geometry_box_indices=(0, 0),
            control_poly_indices=(0, 0),
            traffic_control_element_type=-1,
            traffic_control_element_id=-1,
        )


@dataclass
class RoadSegmentInfo:
    """Metadata for a road segment polygon.

    Attributes:
        poly_indices: Index range into the polygon array: "road_segments".
        rs_id: Unique road segment ID.
    """

    poly_indices: tuple[int, int]
    rs_id: int

    @classmethod
    def create_empty(cls):
        """Describes a crosswalk area in the environment.

        Attributes:
            poly_indices: Index range into the polygon array.
            cw_id: Unique crosswalk ID.
        """
        return RoadSegmentInfo(poly_indices=(0, 0), rs_id=-1)


@dataclass
class CrosswalkInfo:
    """Describes a crosswalk area in the environment.

    Attributes:
        poly_indices: Index range into the polygon array: "crosswalks".
        cw_id: Unique crosswalk ID.
    """

    poly_indices: tuple[int, int]
    cw_id: int

    @classmethod
    def create_empty(cls):
        """Creates an empty CrosswalkInfo with default values."""
        return CrosswalkInfo(poly_indices=(0, 0), cw_id=-1)


@dataclass
class ParkingSpaceInfo:
    """Metadata for a parking space region.

    Attributes:
        poly_indices: Index range into the polygon array: "parking_area".
        ps_id: Unique parking space ID.
    """

    poly_indices: tuple[int, int]
    ps_id: int

    @classmethod
    def create_empty(cls):
        """Creates an empty ParkingSpaceInfo with default values."""
        return ParkingSpaceInfo(poly_indices=(0, 0), ps_id=-1)


@dataclass
class DriveableSpaceInfo:
    """Metadata for a drivable space polygon.

    Attributes:
        poly_indices: Index range into the polygon array: "driveable_space".
        ds_id: Unique identifier for the drivable space.
    """

    poly_indices: tuple[int, int]
    ds_id: int

    @classmethod
    def create_empty(cls):
        """Creates an empty DriveableSpaceInfo with default values."""
        return DriveableSpaceInfo(poly_indices=(0, 0), ds_id=-1)


@dataclass
class LaneLineInfo:
    """Metadata for a lane line.

    Attributes:
        lane_line_data_indices: Index range into the per-point data arrays:
            * "lane_lines"
            * "lane_line_types"
            * "lane_line_left_driving_directions"
            * "lane_line_right_driving_directions"
        ll_id: Unique identifier for the lane line.
    """

    lane_line_data_indices: tuple[int, int]
    ll_id: int

    @classmethod
    def create_empty(cls) -> Self:
        """Creates an empty DriveableSpaceInfo with default values."""
        return cls(lane_line_data_indices=(0, 0), ll_id=-1)


@dataclass
class EpisodeMap:
    """Represents a local map of the environment for a given episode.

    Stores vectorized representations of static world elements (lanes, crosswalks,
    parking areas, driveable space, and traffic control elements) along with
    metadata and connectivity information.

    Attributes:
        lane_centerlines: A tensor of shape (P, 4) describing lane center polylines,
            where P is the total number of points across all lanes.
        lane_left_boundaries: A tensor of shape (P, 4) describing left boundary polylines.
        lane_right_boundaries: A tensor of shape (P, 4) describing right boundary polylines.
        lane_infos: A list of LaneSegmentInfo instances holding metadata for each lane.

        road_segments: A tensor of shape (P, 4) for road segment polygons.
        road_segments_infos: A list of RoadSegmentInfo instances for each road segment.

        crosswalks: A tensor of shape (P, 4) for crosswalk polygons.
        crosswalk_infos: A list of CrosswalkInfo instances for each crosswalk.

        parking_area: A tensor of shape (P, 4) for parking area polygons.
        parking_area_infos: A list of ParkingSpaceInfo instances for each parking space.

        driveable_space: A tensor of shape (P, 4) for drivable space polygons.
        driveable_space_infos: A list of DriveableSpaceInfo instances for each drivable area.

        tce_geometry_boxes: A tensor of shape (P, 4) for traffic control element boxes.
        tce_control_polys: A tensor of shape (P, 4) for traffic control element polygons.
        tce_infos: A list of TrafficControlElementInfo instances for each element.

        lane_id_to_idx: A mapping from lane ID to its index in lane_infos.
        tce_id_to_idx: A mapping from traffic control element ID to its index in tce_infos.
    """

    lane_centerlines: np.ndarray | torch.Tensor
    lane_left_boundaries: np.ndarray | torch.Tensor
    lane_right_boundaries: np.ndarray | torch.Tensor

    lane_infos: list[LaneSegmentInfo]

    road_segments: np.ndarray | torch.Tensor
    road_segments_infos: list[RoadSegmentInfo]

    crosswalks: np.ndarray | torch.Tensor
    crosswalk_infos: list[CrosswalkInfo]

    parking_area: np.ndarray | torch.Tensor
    parking_area_infos: list[ParkingSpaceInfo]

    driveable_space: np.ndarray | torch.Tensor
    driveable_space_infos: list[DriveableSpaceInfo]

    lane_lines: np.ndarray | torch.Tensor
    lane_line_types: np.ndarray | torch.Tensor
    lane_line_left_driving_directions: np.ndarray | torch.Tensor
    lane_line_right_driving_directions: np.ndarray | torch.Tensor
    lane_lines_infos: list[LaneLineInfo]

    lane_id_to_idx: dict[int, int]

    @classmethod
    def create_empty(cls):
        """Creates an empty EpisodeMap with default values."""
        return EpisodeMap(
            lane_centerlines=np.zeros((0, 4), dtype=np.float32),
            lane_left_boundaries=np.zeros((0, 4), dtype=np.float32),
            lane_right_boundaries=np.zeros((0, 4), dtype=np.float32),
            lane_infos=[],
            road_segments=np.zeros((0, 4), dtype=np.float32),
            road_segments_infos=[],
            crosswalks=np.zeros((0, 4), dtype=np.float32),
            crosswalk_infos=[],
            parking_area=np.zeros((0, 4), dtype=np.float32),
            parking_area_infos=[],
            driveable_space=np.zeros((0, 4), dtype=np.float32),
            driveable_space_infos=[],
            lane_lines=np.zeros((0, 4), dtype=np.float32),
            lane_line_types=np.zeros((0,), dtype=int),
            lane_line_left_driving_directions=np.zeros((0,), dtype=int),
            lane_line_right_driving_directions=np.zeros((0,), dtype=int),
            lane_lines_infos=[],
            # tce_geometry_boxes=np.zeros((0, 4), dtype=np.float32),
            # tce_control_polys=np.zeros((0, 4), dtype=np.float32),
            # tce_infos=[],
            # tce_id_to_idx={},
            lane_id_to_idx={},
        )

    ### TODO: This is not nice, needs copy all the time. Add a proper initialization
    ### using list of segments...
    def append_array(self, src, dst) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Append a source array to a destination array with zero-padding to 4 columns,
        setting the 4th column as 1.0 (homogeneous coordinate).

        Args:
            src (np.ndarray): Source array of shape (N, M), M <= 4.
            dst (np.ndarray): Destination array of shape (K, 4).

        Returns:
            tuple[np.ndarray, tuple[int, int]]:
                - The concatenated array.
                - A tuple with start and end indices (slice boundaries) of the appended data in the concatenated array.
        """
        if len(src) == 0:
            return dst, (dst.shape[0], dst.shape[0])

        # Pad src to width 4
        tarr = np.zeros((src.shape[0], 4), dtype=np.float32)
        tarr[:, : src.shape[1]] = src
        start_length = dst.shape[0]
        res = np.concatenate((dst, tarr), axis=0)
        end_length = res.shape[0]
        res[:, 3] = 1.0
        return res, (start_length, end_length)

    def add_segment(self, cl, lb, rb, l_type, lb_type, rb_type, lane_id, speed_limit, is_poly_lane=False):
        """
        Adds a lane segment to the map, appending lane centerlines and boundaries, and updating lane info.

        Args:
            cl (np.ndarray): Centerline points array.
            lb (np.ndarray): Left boundary points array.
            rb (np.ndarray): Right boundary points array.
            l_type (int): Lane type identifier.
            lb_type (int): Left boundary type identifier.
            rb_type (int): Right boundary type identifier.
            lane_id (int): Unique lane identifier.
            speed_limit (float): Speed limit for this lane segment.
            is_poly_lane (bool, optional): Flag indicating if this lane is a polygonal lane. Defaults to False.

        """

        li = LaneSegmentInfo.create_empty()

        self.lane_centerlines, li.centerline_indices = self.append_array(cl, self.lane_centerlines)
        self.lane_left_boundaries, li.left_boundary_indices = self.append_array(lb, self.lane_left_boundaries)
        self.lane_right_boundaries, li.right_boundary_indices = self.append_array(rb, self.lane_right_boundaries)
        li.lane_type = int(l_type)
        li.left_boundary_type = int(lb_type)
        li.right_boundary_type = int(rb_type)
        li.lane_id = lane_id
        li.speed_limit = speed_limit
        li.is_poly_lane = is_poly_lane
        self.lane_infos.append(li)
        if li.lane_id in self.lane_id_to_idx:
            print("Lane " + str(li.lane_id) + " already exists! Lane is not added!!!")
            return
        self.lane_id_to_idx[li.lane_id] = len(self.lane_infos) - 1

    def set_lane_continue(self, src, dst):
        """
        Connects two lanes by setting the destination lane as a successor of the source lane,
        and vice versa as a predecessor.

        Args:
            src (int): Lane ID of the source lane.
            dst (int): Lane ID of the destination lane.

        """
        if self.lane_id_to_idx[dst] not in self.lane_infos[self.lane_id_to_idx[src]].successors:
            self.lane_infos[self.lane_id_to_idx[src]].successors.append(self.lane_id_to_idx[dst])
        if self.lane_id_to_idx[src] not in self.lane_infos[self.lane_id_to_idx[dst]].predecessors:
            self.lane_infos[self.lane_id_to_idx[dst]].predecessors.append(self.lane_id_to_idx[src])

    def add_road_segment(self, poly, rs_id):
        """
        Adds a road segment polygon and stores its ID.

        Args:
            poly (np.ndarray): Polygon points of the road segment.
            rs_id (int): Unique identifier for the road segment.
        """

        info = RoadSegmentInfo.create_empty()
        self.road_segments, info.poly_indices = self.append_array(poly, self.road_segments)
        info.rs_id = rs_id
        self.road_segments_infos.append(info)

    def add_drivable_space(self, poly, ds_id):
        """
        Adds a drivable space polygon and stores its ID.

        Args:
            poly (np.ndarray): Polygon points of the drivable space.
            ds_id (int): Unique identifier for the drivable space.
        """
        info = DriveableSpaceInfo.create_empty()
        self.driveable_space, info.poly_indices = self.append_array(poly, self.driveable_space)
        info.ds_id = ds_id
        self.driveable_space_infos.append(info)

    def add_lane_line(
        self,
        line: npt.NDArray[np.float64],
        types: npt.NDArray[np.int64],
        left_driving_directions: npt.NDArray[np.int64],
        right_driving_directions: npt.NDArray[np.int64],
        ll_id: int,
    ):
        """
        Adds a lane line and stores its ID.

        Args:
            line (np.ndarray): Points of the lane line.
            types (np.ndarray): Type enums of the lane line points.
            left_driving_directions (np.ndarray): Enums of driving direction on the left.
            right_driving_direction (np.ndarray): Enums of driving direction on the right.
            ll_id (int): ID of the lane line.
        """
        info = LaneLineInfo.create_empty()
        info.ll_id = ll_id
        self.lane_lines, info.lane_line_data_indices = self.append_array(line, self.lane_lines)
        self.lane_line_types = np.concatenate((self.lane_line_types, types), axis=0)
        self.lane_line_left_driving_directions = np.concatenate(
            (self.lane_line_left_driving_directions, left_driving_directions), axis=0
        )
        self.lane_line_right_driving_directions = np.concatenate(
            (self.lane_line_right_driving_directions, right_driving_directions), axis=0
        )
        self.lane_lines_infos.append(info)

    def add_crosswalk(self, poly, cw_id):
        """
        Adds a crosswalk polygon and stores its ID.

        Args:
            poly (np.ndarray): Polygon points of the crosswalk.
            cw_id (int): Unique identifier for the crosswalk.

        """
        info = CrosswalkInfo.create_empty()
        self.crosswalks, info.poly_indices = self.append_array(poly, self.crosswalks)
        info.cw_id = cw_id
        self.crosswalk_infos.append(info)

    def add_parking_space(self, poly, ps_id):
        """
        Adds a parking space polygon and stores its ID.

        Args:
            poly (np.ndarray): Polygon points of the parking space.
            ps_id (int): Unique identifier for the parking space.
        """

        info = ParkingSpaceInfo.create_empty()
        self.parking_area, info.poly_indices = self.append_array(poly, self.parking_area)
        info.ps_id = ps_id
        self.parking_area_infos.append(info)

    def get_centerline_by_idx(self, idx):
        """
        Retrieves the centerline points of a lane by its lane ID.

        Args:
            idx (int): Lane ID.

        Returns:
            np.ndarray: Array of centerline points for the specified lane.
        """
        centerline_indices = self.lane_infos[self.lane_id_to_idx[idx]].centerline_indices
        return self.lane_centerlines[centerline_indices[0] : centerline_indices[1]]

    def get_local_crop(
        self,
        reference_frame: np.ndarray | torch.Tensor,
        region_of_interest: list,
    ) -> None:
        """Extract a local cropped EpisodeMap in a specified reference frame.

        Transforms and prunes map elements based on the given 4x4 homogeneous
        transform and axis-aligned bounding box. Implementation is not provided.

        Args:
            reference_frame: 4x4 homogeneous transform (tensor shape: (B>=1, 1, 4, 4)).
            region_of_interest: Bounding box defined as [x_min, y_min, x_max, y_max].

        Returns:
            None

        Raises:
            NotImplementedError: Always raised since functionality is not implemented.
        """
        ### TODO: Define return. Probably an EpisodeMap object with data tensors transformed and pruned accordingly?
        raise NotImplementedError

    def rasterize(
        self,
        grid_resolution: int,
    ) -> None:
        """Rasterize the EpisodeMap into a bird's-eye-view grid.

        Converts vectorized map representations into a raster image given the
        grid resolution in meters per pixel. Implementation is not provided.

        Args:
            grid_resolution: Grid resolution in meters per pixel.

        Returns:
            None

        Raises:
            NotImplementedError: Always raised since functionality is not implemented.
        """
        ### TODO: This is done by the rasterization tools, this function can be removed?
        ### TODO: Define return. Probably an tensor of shape (B>=1, N, H, W, C)
        # representing the rasterized map, with C being lane, crosswalk
        # and parking area features and H and W resulting from the
        # current ROI and grid resolution.
        raise NotImplementedError

    def transform_to(self, m: np.ndarray) -> EpisodeMap:
        """
        Applies a homogeneous transformation matrix to all geometric data in an EpisodeMap.

        This function returns a new `EpisodeMap` instance where all 3D coordinates (e.g.,
        centerlines, boundaries, polygons) have been transformed by the given 4x4 matrix.
        Metadata such as lane IDs and type information are preserved by shallow copy.

        Args:
            emap (EpisodeMap): The source episode map to be transformed.
            m (np.ndarray): A 4x4 homogeneous transformation matrix (typically used for
                translation, rotation, and scaling in 3D space).

        Returns:
            EpisodeMap: A new `EpisodeMap` with transformed geometry.
        """

        res = EpisodeMap.create_empty()

        res.lane_infos = self.lane_infos
        res.lane_id_to_idx = self.lane_id_to_idx
        res.lane_centerlines = self.lane_centerlines @ np.transpose(m)
        res.lane_left_boundaries = self.lane_left_boundaries @ np.transpose(m)
        res.lane_right_boundaries = self.lane_right_boundaries @ np.transpose(m)

        res.road_segments_infos = self.road_segments_infos
        res.road_segments = self.road_segments @ np.transpose(m)

        res.crosswalk_infos = self.crosswalk_infos
        res.crosswalks = self.crosswalks @ np.transpose(m)

        res.driveable_space_infos = self.driveable_space_infos
        res.driveable_space = self.driveable_space @ np.transpose(m)

        res.parking_area_infos = self.parking_area_infos
        res.parking_area = self.parking_area @ np.transpose(m)

        res.lane_infos = self.lane_infos
        res.lane_lines = self.lane_lines @ np.transpose(m)

        return res


@dataclass
class MapContainer(BaseEntity):
    """Container class that holds a batch of `EpisodeMap` instances."""

    maps: list[EpisodeMap]  # = field(default_factory=lambda: [EpisodeMap.create_empty()])

    @property
    def dimensionality(self) -> int:
        return 1

    @property
    def t_dim(self) -> None:
        """Returns the size of the time dimension. Returns none since object has no time dimension."""
        return None

    @property
    def n_dim(self) -> None:
        """Returns the size of the element dimension. Returns none since object has no element dimension."""
        return None

    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        self._check_list_dim("maps", 1, (), ignore_list)

    @classmethod
    def aggregated_time(cls, time: list[MapContainer], use_custom_batching: list | None = None) -> MapContainer:
        """
        Aggregates a list of MapContainer instances over time. We assume map is static and only use the most recent one.

        Args:
            time (list[MapContainer]): List of MapContainer objects over time.
            use_custom_batching: unused, only available to match interface of base class

        Returns:
            MapContainer: The last MapContainer in the list.
        """
        return time[-1]

    @classmethod
    def aggregated_batch(
        cls, batch: list[Self], use_custom_batching: list | None = None, max_t: int = 0, fill_up_to_max: bool = False
    ) -> Self:
        res = super(MapContainer, cls).aggregated_batch(batch, ["maps"])
        res.maps = [m.maps[0] for m in batch]
        return res


class EpisodeMapUtils:
    """Utility functions for working with EpisodeMap instances."""

    @staticmethod
    def generate_connectivity(em: EpisodeMap, dist_threshold=1.0):
        """
        Generates connectivity information (successors and predecessors) between lanes
        in the episode map based on a distance threshold.

        This function assumes that lanes are oriented and attempts to detect lane
        transitions by comparing the end of one centerline to the start of another.

        Args:
            em (EpisodeMap): The episode map containing lane information and geometries.
            dist_threshold (float): The maximum distance between end/start points of
                lanes to consider them connected. Defaults to 1.0.
        """

        ## HACK to make lanes unique
        ## TODO: Check where lanes are duplicated, sub-map extraction?
        # laneset = set(self.lanes)
        # self.lanes = list(laneset)

        for i in range(len(em.lane_infos)):
            lane = em.lane_infos[i]
            lane.successors = []
            lane.predecessors = []

        for src in em.lane_infos:
            for dst in em.lane_infos:
                if src.lane_id == dst.lane_id:
                    continue
                dist = np.linalg.norm(
                    em.lane_centerlines[src.centerline_indices[1] - 1] - em.lane_centerlines[dst.centerline_indices[0]]
                )

                if dist < dist_threshold and dst.lane_id not in src.successors:
                    src.successors.append(dst.lane_id)
                    dst.predecessors.append(src.lane_id)

    @staticmethod
    def get_lane_geometry(em: EpisodeMap, lane_info: LaneSegmentInfo):
        """
        Retrieves the geometric components (centerline and boundaries) of a lane.

        Args:
            em (EpisodeMap): The episode map containing lane geometries.
            lane_info (LaneSegmentInfo): The lane segment for which geometry is queried.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the centerline,
                left boundary, and right boundary as NumPy arrays.
        """
        return (
            em.lane_centerlines[lane_info.centerline_indices[0] : lane_info.centerline_indices[1]],
            em.lane_left_boundaries[lane_info.left_boundary_indices[0] : lane_info.left_boundary_indices[1]],
            em.lane_right_boundaries[lane_info.right_boundary_indices[0] : lane_info.right_boundary_indices[1]],
        )

    @staticmethod
    def get_submap_box(em: EpisodeMap, box_min, box_max) -> EpisodeMap:
        """
        Extracts a submap of the episode map containing only the lanes whose centerlines
        lie (partially) within a specified bounding box.

        This function currently only filters lane segments.

        Args:
        em (EpisodeMap): The original episode map.
        box_min (tuple[float, float]): The (x_min, y_min) coordinates defining the
            lower bound of the bounding box.
        box_max (tuple[float, float]): The (x_max, y_max) coordinates defining the
            upper bound of the bounding box.

        Returns:
            EpisodeMap: A new episode map containing only lane segments within the bounding box.
        """

        res_map = EpisodeMap.create_empty()

        if not hasattr(em.lane_centerlines, "shape"):
            raise AttributeError("EpisodeMap.lane_centerlines must have shape attribute")
        inside = np.ones(em.lane_centerlines.shape[0], dtype=np.uint8)

        inside[em.lane_centerlines[:, 0] < box_min[0]] = 0
        inside[em.lane_centerlines[:, 0] > box_max[0]] = 0
        inside[em.lane_centerlines[:, 1] < box_min[1]] = 0
        inside[em.lane_centerlines[:, 1] > box_max[1]] = 0
        # TODO: implement for crosswalks, road_segments, driveable space, ...
        for li in em.lane_infos:
            if np.max(inside[li.centerline_indices[0] : li.centerline_indices[1]]) > 0:
                cl, lb, rb = EpisodeMapUtils.get_lane_geometry(em, li)
                res_map.add_segment(
                    cl,
                    lb,
                    rb,
                    li.lane_type,
                    li.left_boundary_type,
                    li.right_boundary_type,
                    li.lane_id,
                    li.speed_limit,
                )

        return res_map

    @staticmethod
    def make_2d(em: EpisodeMap):
        """
        Projects all 3D map elements in the episode map onto the 2D plane by zeroing out the Z-axis.

        Args:
            em (EpisodeMap): The episode map whose geometrical components are to be flattened to 2D.

        Modifies:
            - Sets the z-coordinate of the following map components to zero in-place:
                - lane_centerlines
                - lane_left_boundaries
                - lane_right_boundaries
                - road_segments
                - crosswalks
                - driveable_space
                - parking_area
        """
        em.lane_centerlines[:, 2] = 0
        em.lane_left_boundaries[:, 2] = 0
        em.lane_right_boundaries[:, 2] = 0

        em.road_segments[:, 2] = 0
        em.crosswalks[:, 2] = 0
        em.driveable_space[:, 2] = 0
        em.parking_area[:, 2] = 0

    @staticmethod
    def transform(emap: EpisodeMap, m: np.ndarray) -> EpisodeMap:
        """
        Applies a homogeneous transformation matrix to all geometric data in an EpisodeMap.

        This function returns a new `EpisodeMap` instance where all 3D coordinates (e.g.,
        centerlines, boundaries, polygons) have been transformed by the given 4x4 matrix.
        Metadata such as lane IDs and type information are preserved by shallow copy.

        Args:
            emap (EpisodeMap): The source episode map to be transformed.
            m (np.ndarray): A 4x4 homogeneous transformation matrix (typically used for
                translation, rotation, and scaling in 3D space).

        Returns:
            EpisodeMap: A new `EpisodeMap` with transformed geometry.
        """

        res = EpisodeMap.create_empty()

        res.lane_infos = emap.lane_infos
        res.lane_id_to_idx = emap.lane_id_to_idx
        res.lane_centerlines = emap.lane_centerlines @ np.transpose(m)
        res.lane_left_boundaries = emap.lane_left_boundaries @ np.transpose(m)
        res.lane_right_boundaries = emap.lane_right_boundaries @ np.transpose(m)

        res.road_segments_infos = emap.road_segments_infos
        res.road_segments = emap.road_segments @ np.transpose(m)

        res.crosswalk_infos = emap.crosswalk_infos
        res.crosswalks = emap.crosswalks @ np.transpose(m)

        res.driveable_space_infos = emap.driveable_space_infos
        res.driveable_space = emap.driveable_space @ np.transpose(m)

        res.parking_area_infos = emap.parking_area_infos
        res.parking_area = emap.parking_area @ np.transpose(m)

        # res.tce_infos = emap.tce_infos
        # res.tce_id_to_idx = emap.tce_id_to_idx
        # res.tce_geometry_boxes = emap.tce_geometry_boxes @ np.transpose(m)
        # res.tce_control_polys = emap.tce_control_polys @ np.transpose(m)

        return res
