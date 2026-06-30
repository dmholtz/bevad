from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch

from bevad_sim.data_interface.base_entity import BaseEntity

# use typing.TypeAlias starting with py3.10
TorchOrNumpy = Union[np.ndarray, torch.Tensor]


@dataclass
class WorldState(BaseEntity):
    """
    Represents the true state (i.e. privileged) of dynamic agents and traffic
    control elements (lights and signs) for a given scene.

    The overall state is decomposed into several tensors describing all N
    entities under consideration. Each tensor has a batch and time dimension
    equal to or greater than one, meaning it can represent the true state for a
    single frame, a batch of frames, a sequence of frames or both. Each tensor
    represents a different aspect of the state, such as the 4x4 homogenous
    transforms to the center points of each entity, the extent of each entity
    or its dynamics and uses the episode reference frame (ego pose at t=0) by
    default. The ego vehicle always has index 0 in the entity dimension.

    Attributes:
        transform: A tensor of shape (B>=1, T>=1, N, 4, 4) holding the 4x4
            homogenous transforms to the center of each entities bounding box. All
            transformations assume a metric right-hand coordinate system.
        extent: A tensor of shape (B>=1, T>=1, N, 3) holding the extent of each
            entity (all values are full-size dimensions, i.e. length, width, heigth)
        dynamics: A tensor of shape (B>=1, T>=1, N, 4) holding the speed,
            acceleration, yaw rate and TODO: what was it?.
        category: A tensor of shape (B>=1, T>=1, N) holding the semantic
            class of each entity encoded as an integer id.
        attribute: A tensor of shape (B>=1, T>=1, N) holding optional information about
            each entity encoded as an integer, e.g., whether the car is parking or the
            state of a traffic light.
        track_id: A tensor of shape (B>=1, T>=1, N, 4) holding a unique integer
            id for each entity that is consistent across an episode.
        is_valid: A mask tensor of shape (B>=1, T>=1, N) indicating whether
            there is a valid ground truth state for this entity. For example,
            padded or temporarily fully occluded entities can be marked as
            invalid.
        is_visible: A boolean tensor of shape (B>=1, T>=1, N) indicating whether
            each entity is visible in the current frame.
        ego_id: np.ndarray | None  # = ego id
        category_map: A dictionary mapping category names (str) to category ids (int).
        attribute_map: A dictionary mapping attribute names (str) to attribute ids (int).

    """

    transform: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 0, 4, 4), dtype=np.float64))
    extent: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 0, 3), dtype=np.float32))
    dynamics: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 0, 4), dtype=np.float64))
    category: np.ndarray | torch.Tensor | None  # = field(default_factory=lambda: np.zeros((1, 1, 0), dtype=np.uint8))
    attribute: np.ndarray | torch.Tensor | None  # = field(default_factory=lambda: np.zeros((1, 1, 0), dtype=np.uint32))
    track_id: np.ndarray | torch.Tensor  # = field(default_factory=lambda: np.zeros((1, 1, 0), dtype=np.int32))
    is_visible: np.ndarray | torch.Tensor | None
    ego_id: np.ndarray | None  # = field(default_factory=lambda: np.zeros((1,), dtype=np.int32))
    category_map: dict[str, int] | None  # = field(default_factory=dict)
    attribute_map: dict[str, int] | None  # = field(default_factory=dict)

    @property
    def dimensionality(self) -> int:
        return 3

    @property
    def t_dim(self) -> int:
        """Returns the size of the time dimension."""
        return self.is_valid.shape[1]

    @property
    def n_dim(self) -> int:
        """Returns the size of the element dimension."""
        return self.is_valid.shape[2]

    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        self._check_array_dim("transform", 3, (4, 4), ignore_list)
        self._check_array_dim("extent", 3, (3,), ignore_list)
        self._check_array_dim("dynamics", 3, (4,), ignore_list)
        self._check_array_dim("category", 3, None, ignore_list)
        self._check_array_dim("attribute", 3, None, ignore_list)
        self._check_array_dim("track_id", 3, None, ignore_list)
        self._check_array_dim("is_visible", 3, None, ignore_list)
        self._check_array_dim("ego_id", 1, None, ignore_list)

    def to_reference_frame(self, reference_frame: np.ndarray | torch.Tensor) -> WorldState:
        """
        Transforms the world state to a given reference frame.

        Args:
            reference_frame: The reference frame to transform to provided as
                4x4 homogenous transform. Expects a tensor of shape a tensor of
                shape (B>=1, T=1, 4, 4).

        Returns:
            transformed_state: A new WorldState object transformed to the given
            reference frame.
        """
        assert self.transform is not None

        # TODO: @NHANSEL add exact 4x4 homogenous inverse func and use here
        inverse_reference_frame: np.ndarray | torch.Tensor
        if isinstance(reference_frame, torch.Tensor):
            inverse_reference_frame = reference_frame.inverse()
        else:  # backend is numpy
            inverse_reference_frame = np.linalg.inv(reference_frame)

        new_transform = inverse_reference_frame @ self.transform

        return WorldState(
            transform=new_transform,
            extent=self.extent,
            dynamics=self.dynamics,
            category=self.category,
            attribute=self.attribute,
            track_id=self.track_id,
            is_valid=self.is_valid,
            is_visible=self.is_visible,
            ego_id=self.ego_id,
            category_map=self.category_map,
            attribute_map=self.attribute_map,
        )

    def transform_to(self, trans_matrix: np.ndarray | torch.Tensor) -> WorldState:
        """
        Transforms the world state to a given reference frame.

        Args:
            reference_frame: The reference frame to transform to provided as
                4x4 homogenous transform. Expects a tensor of shape a tensor of
                shape (B>=1, T=1, 4, 4).

        Returns:
            transformed_state: A new WorldState object transformed to the given
            reference frame.
        """
        # TODO: @NHANSEL add exact 4x4 homogenous inverse func and use here
        assert self.shape[0] == 1, "For now transforming batched data is not supported! "
        assert self.transform is not None

        new_transform = trans_matrix @ self.transform

        return WorldState(
            is_valid=self.is_valid,
            transform=new_transform,
            extent=self.extent,
            dynamics=self.dynamics,
            category=self.category,
            attribute=self.attribute,
            track_id=self.track_id,
            is_visible=self.is_visible,
            ego_id=self.ego_id,
            category_map=self.category_map,
            attribute_map=self.attribute_map,
        )

    def to_egocentric_frame(self) -> None:
        """
        Transforms the world state to an egocentric frame.
        """
        raise NotImplementedError

    def to_episode_frame(self) -> None:
        """
        Creates an episode frame from the current world state.
        """
        raise NotImplementedError

    @classmethod
    def _create_zeros(cls, b=1, t=1, n=1) -> WorldState:
        """Creates an empty WorldState instance with all fields initialized to zeros.

        Args:
            b (int, optional): Batch size. Defaults to 1.
            t (int, optional): Number of time steps. Defaults to 1.
            n (int, optional): Number of objects. Defaults to 1.

        Returns:
            WorldState: An instance of WorldState with all fields (transform, extent, dynamics, category, attribute, track_id, is_valid)
            initialized to arrays of zeros with appropriate shapes and data types.
        """

        transform = np.zeros((b, t, n, 4, 4), dtype=np.float64)
        extent = np.zeros((b, t, n, 3), dtype=np.float32)
        dynamics = np.zeros((b, t, n, 4), dtype=np.float64)
        category = np.zeros((b, t, n), dtype=np.uint8)
        attribute = np.zeros((b, t, n), dtype=np.int32)
        track_id = np.zeros((b, t, n), dtype=np.int32)
        is_valid = np.zeros((b, t, n), dtype=bool)
        is_visible = np.ones((b, t, n), dtype=bool)

        return WorldState(
            is_valid=is_valid,
            transform=transform,
            extent=extent,
            dynamics=dynamics,
            category=category,
            attribute=attribute,
            track_id=track_id,
            is_visible=is_visible,
            ego_id=None,
            category_map={},
            attribute_map={},
        )

    @classmethod
    def aggregated_time(cls, time: list[WorldState], use_custom_batching: list | None = None) -> WorldState:
        """Aggregates single-batch and single-time world state elements over time.

        Args:
            world_states (list[WorldState]): A chronological list of single-batch, single-timestep
                uninterrupted world state observations.
            align_tracks (bool): If true, the aggregation aligns agent track ids along the agent
                dimension. If false, the aggregation makes the agent dimension compact.

        """

        if len(time) == 0:
            return WorldState._create_zeros(b=1, t=0, n=1)

        return WorldState._aggregated_agent_aligned(time)

    @classmethod
    def aggregate_agents(cls, world_states: list[WorldState]) -> WorldState:
        array_items = ["transform", "extent", "dynamics", "category", "attribute", "track_id", "is_valid", "is_visible"]
        res = WorldState.create_empty()

        for k in array_items:
            res.__dict__[k] = np.concatenate([ws.__dict__[k] for ws in world_states], axis=2)

        res.ego_id = world_states[0].ego_id

        return res

    @classmethod
    def aggregated_time_compact(cls, world_states: list[WorldState]) -> WorldState:
        """
        Aggregates a list of single-batch, single-time `WorldState` objects over time into a compact representation.
        This method combines multiple `WorldState` instances, each representing a single time step and batch,
        into a single `WorldState` object with the time dimension aggregated. The agent (n) dimension is set
        to the maximum number of agents present in any of the input states, and all relevant fields are copied
        over for each time step. The aggregation is not agent-aligned; instead, it compacts the agent dimension
        as much as possible.

        Args:
            world_states (list[WorldState]): A list of `WorldState` objects, each with shape (1, 1, n).

        Returns:
            WorldState: A new `WorldState` object with shape (1, t, max_n), where `t` is the number of input
            states and `max_n` is the maximum agent count across all input states.

        Raises:
            AssertionError: If any input `WorldState` does not have a batch size of 1 or a time dimension of 1.

        Note:
            The use case for this aggregation is unclear; typically, agent alignment may be preferred.
        """
        ### TODO: What is the use case for this? Shouldn't it always be agent aligned?

        t_dim = len(world_states)
        max_n_dim = max(map(lambda ws: ws.shape[2], world_states))

        agg = WorldState._create_zeros(b=1, t=t_dim, n=max_n_dim)
        for t, ws in enumerate(world_states):
            assert ws.shape[1] == 1
            n_dim = ws.shape[2]
            if agg.transform is not None and ws.transform is not None:
                agg.transform[0, t, :n_dim] = ws.transform[0, 0, :n_dim]  # type: ignore[assignment]  # MyPy thinks that left side may be a scalar, which is wrong
            if agg.extent is not None and ws.extent is not None:
                agg.extent[0, t, :n_dim] = ws.extent[0, 0, :n_dim]  # type: ignore[assignment]
            if agg.dynamics is not None and ws.dynamics is not None:
                agg.dynamics[0, t, :n_dim] = ws.dynamics[0, 0, :n_dim]  # type: ignore[assignment]
            if agg.category is not None and ws.category is not None:
                agg.category[0, t, :n_dim] = ws.category[0, 0, :n_dim]  # type: ignore[assignment]
            if agg.attribute is not None and ws.attribute is not None:
                agg.attribute[0, t, :n_dim] = ws.attribute[0, 0, :n_dim]  # type: ignore[assignment]
            agg.track_id[0, t, :n_dim] = ws.track_id[0, 0, :n_dim]  # type: ignore[assignment]
            agg.is_valid[0, t, :n_dim] = ws.is_valid[0, 0, :n_dim]  # type: ignore[assignment]
            if agg.is_visible is not None and ws.is_visible is not None:
                agg.is_visible[0, t, :n_dim] = ws.is_visible[0, 0, :n_dim]  # type: ignore[assignment]
            if agg.category_map is not None and ws.category_map is not None:
                agg.category_map.update(ws.category_map)
            if agg.attribute_map is not None and ws.attribute_map is not None:
                agg.attribute_map.update(ws.attribute_map)
        agg.ego_id = world_states[0].ego_id
        return agg

    @classmethod
    def _aggregated_agent_aligned(cls, world_states: list[WorldState]):
        """Aggregates a sequence of WorldState objects over time, aligning agents by their track IDs.
        This method combines multiple single-batch, single-time WorldState elements into a single
        aggregated WorldState, where agents with the same track ID are aligned at the same index
        across all time steps. The resulting agent dimension (N) may be larger than in individual
        world states, as it includes all unique agents observed across the sequence.

        Args:
            world_states (list[WorldState]): A list of WorldState objects, each representing
                the state of the world at a single time step.

        Returns:
            WorldState: An aggregated WorldState object with time and agent dimensions,
                where each agent is aligned by track ID across all time steps.

        Notes:
            - Assumes that each input WorldState contains agent data in the first batch and time index.
            - The output WorldState will have the same ego_id as the first input WorldState.
        """

        # collect unique track ids, preserving the order of first appearance
        track_ids_dict: OrderedDict[int, None] = OrderedDict()
        for ws in world_states:
            track_ids_dict.update({tid: None for tid in ws.track_id[0, 0]})
        track_ids = np.array([tid for tid in track_ids_dict.keys()], dtype=int)

        track_id_look_up = {}
        for i, fid in enumerate(track_ids):
            track_id_look_up[fid] = i

        res = WorldState._create_zeros(t=len(world_states), n=len(track_ids))

        for t, tws in enumerate(world_states):
            for j in range(tws.track_id.shape[2]):
                track_id = tws.track_id[0, 0, j]

                idx = track_id_look_up[track_id]

                if res.transform is not None and tws.transform is not None:
                    res.transform[0, t, idx] = tws.transform[0, 0, j]
                if res.dynamics is not None and tws.dynamics is not None:
                    res.dynamics[0, t, idx] = tws.dynamics[0, 0, j]
                if res.extent is not None and tws.extent is not None:
                    res.extent[0, t, idx] = tws.extent[0, 0, j]
                res.is_valid[0, t, idx] = tws.is_valid[0, 0, j]
                if res.is_visible is not None and tws.is_visible is not None:
                    res.is_visible[0, t, idx] = tws.is_visible[0, 0, j]
                res.track_id[0, t, idx] = track_id
                if res.category is not None and tws.category is not None:
                    res.category[0, t, idx] = tws.category[0, 0, j]
        res.ego_id = world_states[0].ego_id
        res.attribute_map = world_states[0].attribute_map
        res.category_map = world_states[0].category_map
        return res

    def squeeze_invalid(self) -> WorldState:
        """
        Returns a clone of this WorldState instance with all invalid items removed.
        This method filters out items marked as invalid, returning a new WorldState
        containing only valid items. The operation is only defined when both
        batch and time dimensions are equal to 1.
        Returns:
            WorldState: A new WorldState instance containing only valid items.
        Raises:
            AssertionError: If batch dimension is not 1.
            AssertionError: If time dimension is not 1.
        """

        assert self.shape[0] == 1, f"op is not defined for batch dimension={self.shape[0]}"
        assert self.shape[1] == 1, f"op is not defined for time dimension={self.shape[1]}"

        valid_mask = self.is_valid[0, 0] == 1
        return self[0:1, 0:1, valid_mask]


class WorldStateUtils:
    """Utility functions for manipulating and querying WorldState objects.

    This class provides static methods to perform various operations on WorldState
    instances, such as retrieving agent indices, transforming state representations,
    generating 2D/3D bounding boxes, interpolating states over time, and extracting
    specific agent states.

    Methods:
        get_idx_by_id(ws, agent_id, b=0, t=0):
            Returns the index of the agent with the specified ID in the WorldState.

        make_2d(ws):
            Converts the WorldState's transform to a 2D representation.

        get_batched_2d_box_for_agents(ws):
            Computes batched 2D bounding boxes for all agents in the WorldState.

        get_batched_3d_box_for_agents(ws):
            Computes batched 3D bounding boxes for all agents in the WorldState.

        get_state_by_idx(ws, b=0, t=0, n=0):
            Retrieves the transform state for a specific agent by index.

        get_state_2d_by_id(ws, agent_id, b=0, t=0):
            Retrieves the 2D state (x, y, yaw) for an agent by ID.

        transform(ws, m):
            Applies a transformation matrix to the WorldState's transforms.

        interpolate(ws, timestamps, target_delta_t):
            Interpolates the WorldState temporally to a new time resolution.

    Note:
        All methods are static and operate directly on the provided WorldState object.
    """

    @staticmethod
    def get_idx_by_id(ws: WorldState, agent_id: int, b: int = 0, t: int = 0):
        """
        Returns the index of the specified agent ID in the WorldState's track_id array.

        Args:
            ws (WorldState): The world state object containing the track_id array.
            agent_id (int): The ID of the agent to search for.
            b (int, optional): The batch index. Defaults to 0.
            t (int, optional): The time index. Defaults to 0.

        Returns:
            int: The index of the agent with the specified ID.

        Raises:
            IndexError: If the agent_id is not found in the specified batch and time.
        """

        return np.where(ws.track_id[b, t] == agent_id)[0][0].item()

    @staticmethod
    def make_2d(ws: WorldState):
        """
        Converts the transform of a WorldState object to a 2D representation.
        This function asserts that the input WorldState's transform is not batched (i.e., has a batch size of 1),
        extracts the x, y, and yaw components from the transform, and then reconstructs the transform using only
        these 2D components. The updated transform is then assigned back to the WorldState object.

        Args:
            ws (WorldState): The WorldState object whose transform will be converted to 2D.

        Raises:
            AssertionError: If the input transform is batched (i.e., shape[0] != 1).
        """

        assert ws.transform is not None
        assert ws.transform.shape[0] == 1

        ## TODO: should it work on batched data?
        x, y, yaw = TransformsOperations.get_xyyaw_from_transforms(ws.transform)
        res = TransformsOperations.get_transforms_from_xyyaw(x, y, yaw)
        ws.transform = res

    @staticmethod
    def get_batched_2d_box_for_agents(ws: WorldState):
        """
        Computes the batched 2D bounding box corners for agents in the given world state.
        This function calculates the four corners of the 2D bounding box for each agent,
        applies the corresponding transformation, and returns the resulting coordinates.

        Args:
            ws (WorldState): The world state object containing the extent and transformation
                matrices for each agent.

        Returns:
            np.ndarray: An array of shape (..., 4, 2) containing the transformed 2D coordinates
                of the four bounding box corners for each agent.
        """
        assert ws.extent is not None
        assert ws.transform is not None

        return WorldStateUtils.get_batched_2d_boxes(ws.extent, ws.transform)

    @staticmethod
    def get_batched_2d_boxes(ext: TorchOrNumpy, transform: TorchOrNumpy) -> np.ndarray:
        """
        Computes the batched 2D bounding box corners for agents in the given world state.
        This function calculates the four corners of the 2D bounding box for each agent,
        applies the corresponding transformation, and returns the resulting coordinates.

        Args:
            ws (WorldState): The world state object containing the extent and transformation
                matrices for each agent.

        Returns:
            np.ndarray: An array of shape (..., 4, 2) containing the transformed 2D coordinates
                of the four bounding box corners for each agent.
        """

        exth = ext / 2.0

        nshape = list(ext.shape[:-1]) if len(ext.shape) > 1 else []
        nshape.append(4)
        nshape.append(4)

        points = np.zeros(nshape, dtype=np.float32)

        points[..., 0, 0] = -exth[..., 0]
        points[..., 0, 1] = -exth[..., 1]
        points[..., 0, 3] = 1.0

        points[..., 1, 0] = -exth[..., 0]
        points[..., 1, 1] = exth[..., 1]
        points[..., 1, 3] = 1.0

        points[..., 2, 0] = exth[..., 0]
        points[..., 2, 1] = exth[..., 1]
        points[..., 2, 3] = 1.0

        points[..., 3, 0] = exth[..., 0]
        points[..., 3, 1] = -exth[..., 1]
        points[..., 3, 3] = 1.0

        transform_np = transform if isinstance(transform, np.ndarray) else transform.numpy()
        trans = np.moveaxis(transform_np, -1, -2)
        tp = points @ trans
        boxes = tp[..., :2]
        return boxes

    @staticmethod
    def get_batched_3d_box_for_agents(ws: WorldState) -> np.ndarray:
        """
        Computes the batched 3D bounding box corner points for agents in the world state.
        This function generates the 8 corner points (in homogeneous coordinates) of 3D bounding boxes
        for each agent, based on their extents and transformation matrices provided in the given
        `WorldState` object. The resulting points are transformed into world coordinates.

        Args:
            ws (WorldState): The world state containing agent extents and transformation matrices.
                - ws.extent: Array-like, shape (..., 3), representing the size of each agent's box.
                - ws.transform: Array-like, shape (..., 4, 4), representing transformation matrices.

        Returns:
            np.ndarray: The transformed 3D bounding box corner points for each agent, shape (..., 8, 4).
                Each box is represented by 8 corner points in homogeneous coordinates.
        """
        assert ws.extent is not None
        assert ws.transform is not None

        ext = ws.extent
        exth = ext / 2.0

        nshape = list(ext.shape[:-1]) if len(ext.shape) > 1 else []
        nshape.append(8)
        nshape.append(4)

        points = np.zeros(nshape, dtype=np.float64)

        points[..., 0, 0] = -exth[..., 0]
        points[..., 0, 1] = -exth[..., 1]
        points[..., 0, 2] = -exth[..., 2]
        points[..., 0, 3] = 1.0

        points[..., 1, 0] = -exth[..., 0]
        points[..., 1, 1] = -exth[..., 1]
        points[..., 1, 2] = exth[..., 2]
        points[..., 1, 3] = 1.0

        points[..., 2, 0] = -exth[..., 0]
        points[..., 2, 1] = exth[..., 1]
        points[..., 2, 2] = exth[..., 2]
        points[..., 2, 3] = 1.0

        points[..., 3, 0] = -exth[..., 0]
        points[..., 3, 1] = exth[..., 1]
        points[..., 3, 2] = -exth[..., 2]
        points[..., 3, 3] = 1.0

        points[..., 4, 0] = exth[..., 0]
        points[..., 4, 1] = -exth[..., 1]
        points[..., 4, 2] = -exth[..., 2]
        points[..., 4, 3] = 1.0

        points[..., 5, 0] = exth[..., 0]
        points[..., 5, 1] = -exth[..., 1]
        points[..., 5, 2] = exth[..., 2]
        points[..., 5, 3] = 1.0

        points[..., 6, 0] = exth[..., 0]
        points[..., 6, 1] = exth[..., 1]
        points[..., 6, 2] = exth[..., 2]
        points[..., 6, 3] = 1.0

        points[..., 7, 0] = exth[..., 0]
        points[..., 7, 1] = exth[..., 1]
        points[..., 7, 2] = -exth[..., 2]
        points[..., 7, 3] = 1.0

        transform_np = ws.transform if isinstance(ws.transform, np.ndarray) else ws.transform.numpy()
        trans = np.moveaxis(transform_np, -1, -2).astype(np.float64)
        tp = points @ trans

        return tp.astype(np.float32)

    @staticmethod
    def get_state_by_idx(ws: WorldState, b: int = 0, t: int = 0, n: int = 0) -> np.ndarray | torch.Tensor:
        """
        Retrieves the state from the WorldState object at the specified indices.

        Args:
            ws (WorldState): The WorldState instance containing the state data.
            b (int, optional): The batch index. Defaults to 0.
            t (int, optional): The time index. Defaults to 0.
            n (int, optional): The entity index. Defaults to 0.

        Returns:
            Any: The state at the specified indices within the WorldState's transform attribute.
        """
        assert ws.transform is not None

        return ws.transform[b, t, n]

    @staticmethod
    def get_state_2d_by_id(ws: WorldState, agent_id: int, b: int = 0, t: int = 0):
        """
        Retrieves the 2D state (x, y, yaw) of an agent by its ID from the world state.

        Args:
            ws (WorldState): The world state object containing agent information.
            agent_id (int): The unique identifier of the agent.
            b (int, optional): The batch index. Defaults to 0.
            t (int, optional): The time index. Defaults to 0.

        Returns:
            Tuple[float, float, float]: The (x, y, yaw) values representing the agent's 2D state.

        Raises:
            ValueError: If the agent ID is not found in the world state.
        """
        assert ws.transform is not None

        agent_idx = WorldStateUtils.get_idx_by_id(ws, agent_id, b=b, t=t)
        return TransformsOperations.get_xyyaw_from_transforms(ws.transform[b, t, agent_idx])

    @staticmethod
    def get_state_2d_by_idx(ws: WorldState, b: int = 0, t: int = 0, n: int = 0):
        assert ws.dynamics is not None

        x, y, yaw = TransformsOperations.get_xyyaw_from_transforms(WorldStateUtils.get_state_by_idx(ws, b=b, t=t, n=n))
        v = ws.dynamics[b, t, n, 0]
        return np.array([x, y, yaw, v], dtype=np.float32)

    @staticmethod
    def transform(ws: WorldState, m: np.ndarray):
        """
        Applies a transformation matrix to the WorldState's transformation matrix.

        Args:
            ws (WorldState): The world state object containing a transformation matrix.
            m (np.ndarray): A transformation matrix to be applied.

        Returns:
            np.ndarray: The result of multiplying the input matrix with the WorldState's transformation matrix.
        """

        return m @ ws.transform

    @staticmethod
    def interpolate(ws: WorldState, timestamps: np.ndarray, target_delta_t: float) -> tuple[WorldState, np.ndarray]:
        """Interpolates the temporal data of a WorldState object to a new set of timestamps with a specified interval.

        This function performs temporal interpolation of the attributes of a WorldState instance, such as dynamics,
        extent, validity, category, track_id, and transformation matrices, based on the provided original timestamps.
        The interpolation is performed for each track_id, and the resulting WorldState is aligned to a new set of
        timestamps generated with the specified target delta time.

        Args
            ws (WorldState): The input WorldState object containing the data to be interpolated.
            timestamps (np.ndarray): Array of original timestamps corresponding to the data in `ws`.
            target_delta_t (float): The desired time interval (in seconds) between consecutive frames in the output.

            Tuple[WorldState, np.ndarray]:
                - Interpolated WorldState object with data aligned to the new timestamps.
                - Array of new timestamps generated with the specified interval.

        Returns:
            interpolated_ws: Interpolated WorldState list
            new_timestamps: New timestamps array

        Raises:
            ValueError: If the input timestamps array is empty or not monotonically increasing.
        """
        assert ws.transform is not None

        res = WorldState.create_empty()

        # Normalize timestamps to start from 0
        src_timestamps = timestamps.copy()
        src_timestamps = src_timestamps - src_timestamps[0]

        # Generate target timestamps
        seq_length = src_timestamps[-1]

        num_frames = int(np.floor(seq_length / target_delta_t))
        target_timestamps = np.linspace(0, target_delta_t * num_frames, num_frames + 1)

        x, y, z, r, p, yaw = TransformsOperations.xyz_rpy_from_transforms(ws.transform)

        res.dynamics = interpolate_array_time(src_timestamps, ws.dynamics, target_timestamps, 1)
        res.extent = interpolate_array_time(src_timestamps, ws.extent, target_timestamps, 1)
        res.is_valid = interpolate_array_time(src_timestamps, ws.is_valid, target_timestamps, 1).astype(np.uint8)

        res.category = interpolate_array_time(src_timestamps, ws.category, target_timestamps, 1).astype(np.uint16)
        res.track_id = interpolate_array_time(src_timestamps, ws.track_id, target_timestamps, 1).astype(np.uint16)

        tx = interpolate_array_time(src_timestamps, x, target_timestamps, 1)
        ty = interpolate_array_time(src_timestamps, y, target_timestamps, 1)
        tz = interpolate_array_time(src_timestamps, z, target_timestamps, 1)
        tr = interpolate_array_time(src_timestamps, r, target_timestamps, 1)
        tp = interpolate_array_time(src_timestamps, p, target_timestamps, 1)
        tyaw = interpolate_array_time(src_timestamps, yaw, target_timestamps, 1)

        res.transform = TransformsOperations.get_transforms_from_xyz_rpy(tx, ty, tz, tr, tp, tyaw)

        return res, target_timestamps


def interpolate_array_time(x_org, y_org, x_target, time_dim):
    """
    Interpolates values of an array along a specified time dimension.
    This function performs 1D linear interpolation of the input array `y_org`
    along the time axis defined by `x_org` and `x_target`. It supports arrays
    with multiple trailing dimensions, recursively interpolating along the
    last dimension if necessary.

    Args:
        x_org (np.ndarray): 1D array of original time points corresponding to `y_org`.
        y_org (np.ndarray): Array of values to interpolate. The time dimension is
            assumed to be at position `-time_dim`.
        x_target (np.ndarray): 1D array of target time points for interpolation.
        time_dim (int): Number of leading dimensions before the time dimension.

    Returns:
        np.ndarray: Interpolated array with the same shape as `y_org`, except
        the time dimension is replaced by the length of `x_target`.

    Raises:
        ValueError: If the input arrays have incompatible shapes.
    """

    if len(y_org.shape) - time_dim == 1:
        vals = np.interp(x_target, x_org, y_org.reshape(y_org.shape[-1]))
        for i in range(time_dim):
            vals = np.expand_dims(vals, 0)
        return vals
    else:
        res = []
        for i in range(y_org.shape[-1]):
            vals = interpolate_array_time(x_org, y_org[..., i], x_target, time_dim)
            res.append(vals)
        resa = np.stack(res, axis=len(y_org.shape) - 1)
        return resa


def transform_from_2d_state(x: float, y: float, yaw: float):
    """
    Creates a 3D homogeneous transformation matrix from 2D pose (x, y, yaw).

    The resulting 4x4 matrix represents a transformation in 3D space, where the translation is in the XY-plane
    and the rotation is about the Z-axis (yaw).

    Args:
        x (float): The x-coordinate of the translation.
        y (float): The y-coordinate of the translation.
        yaw (float): The rotation angle around the Z-axis, in radians.

    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix as a NumPy array of dtype float64.
    """
    ### TODO: check where used and replce with function from transform operations below

    cy = np.cos(yaw)
    sy = np.sin(yaw)
    r = np.array(
        [[cy, -sy, 0.0, 0.0], [sy, cy, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
    )
    t = np.array([[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    return t @ r


def invert_transform_old(m: np.ndarray):
    ### deprecated, remove
    """
    Inverts a 4x4 homogeneous transformation matrix.
    This function computes the inverse of a transformation matrix that represents
    a rigid body transformation in 3D space (rotation and translation). The input
    matrix is assumed to be a 4x4 NumPy array with the upper-left 3x3 block as the
    rotation matrix and the upper-right 3x1 vector as the translation.

    Args:
        m (np.ndarray): A 4x4 NumPy array representing the transformation matrix.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the inverse transformation matrix.

    Raises:
        ValueError: If the input matrix is not of shape (4, 4).
    """

    irot = np.transpose(m[:3, :3])
    ip = -(irot @ m[:3, 3])
    res = np.zeros_like(m)
    res[:3, :3] = irot
    res[:3, 3] = ip
    res[3, 3] = 1.0
    return res


def inverse_transform_from_pos_rot(pos, rot):
    """
    Computes the inverse of a transformation matrix constructed from position and rotation.
    Given a position vector and a rotation vector (in roll, pitch, yaw), this function computes
    the inverse of the corresponding 4x4 homogeneous transformation matrix.

    Args:
        pos (array-like): A sequence of 3 floats representing the position (x, y, z).
        rot (array-like): A sequence of 3 floats representing the rotation in radians (roll, pitch, yaw).

    Returns:
        numpy.ndarray: A 4x4 numpy array representing the inverse transformation matrix.
    """

    r = TransformsOperations.rotation_matrix_from_rpy(rot[0], rot[1], rot[2])  # 4x4
    t = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)  # 4x1
    r_inv = r.T
    r_inv[:3, 3] -= (r.T @ t)[:3]
    return r_inv


def normalize_angle(angle: float):
    """
    Normalizes an angle to the range [0, 2π).
    This function takes an angle in radians and normalizes it to the interval [0, 2π).

    Args:
        angle (float): The angle in radians to normalize.

    Returns:
        float: The normalized angle in the range [0, 2π).
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def box_from_box_extents(box_extents: np.ndarray):
    """
    Creates the 8 vertices of an axis-aligned box given its extents.

    The function computes the coordinates of the 8 corners of a box centered at the origin,
    with the provided extents along the x, y, and z axes. The result is returned as a
    (8, 4) NumPy array, where each row represents a vertex in homogeneous coordinates (x, y, z, 1).

    Args:
        box_extents (np.ndarray): A 1D array-like of shape (3,) specifying the size of the box
            along the x, y, and z axes (x_size, y_size, z_size).

    Returns:
        np.ndarray: An array of shape (8, 4) containing the coordinates of the 8 box vertices
            in homogeneous coordinates.
    """
    vx = np.array([1.0, 0.0, 0.0])
    vy = np.array([0.0, 1.0, 0.0])
    vz = np.array([0.0, 0.0, 1.0])

    be = box_extents * 0.5
    points = []

    points.append(vx * be[0] + vy * be[1] + vz * be[2])
    points.append(vx * be[0] + -vy * be[1] + vz * be[2])
    points.append(-vx * be[0] + -vy * be[1] + vz * be[2])
    points.append(-vx * be[0] + vy * be[1] + vz * be[2])

    points.append(vx * be[0] + vy * be[1] + -vz * be[2])
    points.append(vx * be[0] + -vy * be[1] + -vz * be[2])
    points.append(-vx * be[0] + -vy * be[1] + -vz * be[2])
    points.append(-vx * be[0] + vy * be[1] + -vz * be[2])

    return np.concatenate((np.array(points), np.ones((len(points), 1))), axis=1)


def box2d_from_box_extents(box_extents: np.ndarray):
    """
    Creates a 4-vertex 2D box (rectangle) in 3D space from the given box extents.
    The function takes the extents (x_size, y_size) of a box and returns the coordinates
    of its four corners as points in 3D space, with the z-coordinate set to 0. The output
    is a NumPy array of shape (4, 4), where each row represents a corner in homogeneous
    coordinates (x, y, z, 1).

    Args:
        box_extents (np.ndarray): A 2-element array-like object specifying the size of the box
            along the x and y axes (x_size, y_size).

    Returns:
        np.ndarray: A (4, 4) array containing the coordinates of the four box corners in
            homogeneous 3D space.
    """
    vx = np.array([1.0, 0.0, 0.0])
    vy = np.array([0.0, 1.0, 0.0])

    be = box_extents * 0.5
    points = []

    points.append(vx * be[0] + vy * be[1])
    points.append(vx * be[0] + -vy * be[1])
    points.append(-vx * be[0] + -vy * be[1])
    points.append(-vx * be[0] + vy * be[1])

    return np.concatenate((np.array(points), np.ones((len(points), 1))), axis=1)


def poly2d_from_transform_extent(transform: np.ndarray, extent: np.ndarray):
    """
    Computes the 2D polygon vertices of a rectangle after applying a transformation.
    Given a 4x4 transformation matrix and an extent (width, height), this function
    creates a rectangle centered at the origin, applies the transformation, and returns
    the resulting 2D coordinates of its four corners.

    Args:
        transform (np.ndarray): A 4x4 transformation matrix to be applied to the rectangle.
        extent (np.ndarray): A 2-element array representing the width and height of the rectangle.

    Returns:
        np.ndarray: A (4, 2) array containing the 2D coordinates of the transformed rectangle's corners.
    """

    eh = extent * 0.5

    box = np.array(
        [[-eh[0], -eh[1], 0.0, 1.0], [eh[0], -eh[1], 0.0, 1.0], [eh[0], eh[1], 0.0, 1.0], [-eh[0], eh[1], 0.0, 1.0]],
        dtype=np.float64,
    )

    return box.dot(transform.T)[:, :2]


def transform_2d_seq(seq, trans_matrix_transposed):
    """
    Applies a 2D affine transformation to a sequence of points.
    This function multiplies the input sequence of 2D points by the rotation/scale part of the
    provided transformation matrix and adds the translation component. The transformation matrix
    is expected to be transposed.

    Args:
        seq (np.ndarray): An array of shape (N, 2) representing N 2D points.
        trans_matrix_transposed (np.ndarray): A 3x3 affine transformation matrix (transposed).

    Returns:
        np.ndarray: The transformed sequence of 2D points with shape (N, 2).
    """
    ### TODO: deprecated, where should it be used? can't it be done in 3D?

    return seq @ trans_matrix_transposed[:2, :2] + trans_matrix_transposed[:2, 2]


class TransformsOperations:
    """Provides a collection of class methods for performing operations on homogeneous transformation matrices,
    typically of shape [B x T x N x 4 x 4]. These operations include extracting position and orientation
    (roll, pitch, yaw) from transformation matrices, constructing transformation matrices from position and
    orientation parameters, and generating rotation or translation matrices.

    Methods:
        xyz_rpy_from_transforms(m: np.ndarray):
            Extracts x, y, z position and roll, pitch, yaw angles from a transformation matrix or array of matrices.

        get_xyyaw_from_transforms(m: np.ndarray):
            Extracts x, y position and yaw angle from a transformation matrix or array of matrices.

        get_transforms_from_xyyaw(x: np.ndarray, y: np.ndarray, yaw: np.ndarray):
            Constructs a 4x4 transformation matrix or array of matrices from x, y position and yaw angle.

        rotation_matrix_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
            Constructs a 4x4 rotation matrix or array of matrices from roll, pitch, and yaw angles.

        translation_matrix_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray):
            Constructs a 4x4 translation matrix or array of matrices from x, y, z position.

        get_transforms_pos_rot(pos: np.ndarray, rot: np.ndarray):
            Constructs a 4x4 transformation matrix or array of matrices from position and rotation vectors.

        get_transforms_from_xyz_rpy(x: np.ndarray, y: np.ndarray, z: np.ndarray, r: np.ndarray, p: np.ndarray, yaw: np.ndarray):
            Constructs a 4x4 transformation matrix or array of matrices from x, y, z position and roll, pitch, yaw angles.
    """

    @classmethod
    def xyz_rpy_from_transforms(
        cls, m: TorchOrNumpy
    ) -> tuple[TorchOrNumpy, TorchOrNumpy, TorchOrNumpy, TorchOrNumpy, TorchOrNumpy, TorchOrNumpy]:
        """
        Extracts translation (x, y, z) and rotation (roll, pitch, yaw) from a transformation matrix.
        This method assumes the input is a 4x4 homogeneous transformation matrix (or a batch of such matrices)
        and computes the corresponding translation and Euler angles (in radians) using the XYZ convention.

        Args:
            m (np.ndarray): A 4x4 numpy array or a batch of such arrays representing transformation matrices.

        Returns:
            Tuple: A tuple containing the translation components (x, y, z) and the rotation components
                (roll, pitch, yaw) in radians. Each component may be a scalar or an array, depending on the input shape.

        Raises:
            ValueError: If the input matrix does not have the correct shape.
        """

        y = np.arctan2(m[..., 1, 0], m[..., 0, 0])
        r = np.arctan2(m[..., 2, 1], m[..., 2, 2])
        p = np.arctan2(-m[..., 2, 0], np.sqrt(m[..., 2, 1] * m[..., 2, 1] + m[..., 2, 2] * m[..., 2, 2]))
        return m[..., 0, 3], m[..., 1, 3], m[..., 2, 3], r, p, y

    @staticmethod
    def get_xyyaw_from_transforms(m: TorchOrNumpy) -> tuple[TorchOrNumpy, TorchOrNumpy, TorchOrNumpy]:
        """Extracts the x, y coordinates and yaw angle from a 3D transformation matrix.

        Args:
            m (np.ndarray): A 3D transformation matrix or a batch of such matrices.
                The expected shape is (..., 4, 4), where the last two dimensions represent
                the transformation matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - x (np.ndarray): The x-coordinate(s) extracted from the transformation(s).
                - y (np.ndarray): The y-coordinate(s) extracted from the transformation(s).
                - yaw (np.ndarray): The yaw angle(s) (rotation around the z-axis) in radians.
        """

        yaw = np.arctan2(m[..., 1, 0], m[..., 0, 0])
        x = m[..., 0, 3]
        y = m[..., 1, 3]
        return x, y, yaw

    @staticmethod
    def get_transforms_from_xyyaw(x: TorchOrNumpy, y: TorchOrNumpy, yaw: TorchOrNumpy) -> TorchOrNumpy:
        """Generates homogeneous transformation matrices from arrays of x, y positions and yaw angles.
        This method constructs 4x4 transformation matrices representing 2D rigid body transformations
        (translation and rotation about the z-axis) for each set of input coordinates and yaw angles.

        Args:
            x (np.ndarray): Array of x positions.
            y (np.ndarray): Array of y positions.
            yaw (np.ndarray): Array of yaw angles in radians.

        Returns:
            np.ndarray: Array of 4x4 homogeneous transformation matrices with shape (..., 4, 4),
                where the leading dimensions match the broadcasted shape of the input arrays.
        """
        if len(x.shape) == 0:
            shape = [4, 4]
        else:
            shape = list(x.shape) + [4, 4]

        rot = np.zeros(shape, dtype=np.float64)
        trans = np.zeros(shape, dtype=np.float64)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        rot[..., 0, 0] = cy
        rot[..., 0, 1] = -sy
        rot[..., 1, 1] = cy
        rot[..., 1, 0] = sy
        rot[..., 2, 2] = 1.0
        rot[..., 3, 3] = 1.0

        trans[..., 0, 3] = x
        trans[..., 1, 3] = y
        trans[..., 0, 0] = 1.0
        trans[..., 1, 1] = 1.0
        trans[..., 2, 2] = 1.0
        trans[..., 3, 3] = 1.0

        return trans @ rot

    @staticmethod
    def rotation_matrix_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
        """Creates a 4x4 rotation matrix from roll, pitch, and yaw angles (in radians).
        The rotation is applied in the order: roll (X axis), pitch (Y axis), then yaw (Z axis).
        The function supports both scalar and array inputs for batch computation.

        Args:
            roll (np.ndarray): Roll angle(s) in radians.
            pitch (np.ndarray): Pitch angle(s) in radians.
            yaw (np.ndarray): Yaw angle(s) in radians.

        Returns:
            np.ndarray: The resulting 4x4 rotation matrix or matrices. The output shape is
                (..., 4, 4), where ... matches the broadcasted shape of the input angles.
        """
        if len(roll.shape) == 0:
            shape = [4, 4]
        else:
            shape = list(roll.shape) + [4, 4]
        # Calculate rotation matrix components
        rot = np.zeros(shape, dtype=np.float64)

        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        rot[..., 0, 0] = cy * cp
        rot[..., 0, 1] = cy * sp * sr - sy * cr
        rot[..., 0, 2] = cy * sp * cr + sy * sr

        rot[..., 1, 0] = sy * cp
        rot[..., 1, 1] = sy * sp * sr + cy * cr
        rot[..., 1, 2] = sy * sp * cr - cy * sr

        rot[..., 2, 0] = -sp
        rot[..., 2, 1] = cp * sr
        rot[..., 2, 2] = cp * cr

        rot[..., 3, 3] = 1.0

        return rot

    @staticmethod
    def translation_matrix_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Creates a translation matrix (or batch of matrices) from x, y, z translation components.

        Args:
            x (np.ndarray): The translation along the x-axis. Can be a scalar or an array.
            y (np.ndarray): The translation along the y-axis. Must be broadcastable to the shape of `x`.
            z (np.ndarray): The translation along the z-axis. Must be broadcastable to the shape of `x`.

        Returns:
            np.ndarray: A 4x4 translation matrix (or batch of matrices) with the translation components set to (x, y, z).
                The shape is (..., 4, 4), where ... is the broadcasted shape of the input arrays.

        """
        # Calculate rotation matrix components
        if len(x.shape) == 0:
            shape = [4, 4]
        else:
            shape = list(x.shape) + [4, 4]
        trans = np.zeros(shape, dtype=np.float64)

        trans[..., 0, 3] = x
        trans[..., 1, 3] = y
        trans[..., 2, 3] = z

        trans[..., 0, 0] = 1.0
        trans[..., 1, 1] = 1.0
        trans[..., 2, 2] = 1.0
        trans[..., 3, 3] = 1.0
        return trans

    @staticmethod
    def get_transforms_pos_rot(pos: np.ndarray, rot: np.ndarray):
        """Creates a transformation matrix from position and rotation vectors.

        Args:
            pos (np.ndarray): Position array with shape (..., 3), representing [x, y, z] coordinates.
            rot (np.ndarray): Rotation array with shape (..., 3), representing [roll, pitch, yaw] in radians.

        Returns:
            np.ndarray: The resulting transformation matrix or matrices.

        """
        return TransformsOperations.get_transforms_from_xyz_rpy(
            pos[..., 0], pos[..., 1], pos[..., 2], rot[..., 0], rot[..., 1], rot[..., 2]
        )

    @staticmethod
    def get_transforms_from_xyz_rpy(
        x: np.ndarray, y: np.ndarray, z: np.ndarray, r: np.ndarray, p: np.ndarray, yaw: np.ndarray
    ):
        """
        Creates a transformation matrix from position (x, y, z) and orientation (roll, pitch, yaw).

        Args:
            x (np.ndarray): The x-coordinate(s) of the translation.
            y (np.ndarray): The y-coordinate(s) of the translation.
            z (np.ndarray): The z-coordinate(s) of the translation.
            r (np.ndarray): The roll angle(s) in radians.
            p (np.ndarray): The pitch angle(s) in radians.
            yaw (np.ndarray): The yaw angle(s) in radians.

        Returns:
            np.ndarray: The resulting transformation matrix or matrices combining translation and rotation.
        """
        t = TransformsOperations.translation_matrix_from_xyz(x, y, z)
        r = TransformsOperations.rotation_matrix_from_rpy(r, p, yaw)
        return t @ r
