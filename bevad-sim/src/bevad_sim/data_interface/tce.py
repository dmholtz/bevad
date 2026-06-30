from __future__ import annotations

from dataclasses import dataclass
from typing import List, Type, TypeVar

import numpy as np
import torch

from bevad_sim.data_interface.base_entity import BaseEntity
from bevad_sim.data_interface.data_types import TrafficLightState

TCE = TypeVar("TCE", bound="TrafficControlElements")


@dataclass
class TrafficControlElements(BaseEntity):
    """Represents the privileged state of traffic control elements such as traffic lights and traffic signs.
    This class encapsulates the batched and time-sequenced properties of traffic control elements,
    including their spatial transformations, extents, categories, unique identifiers, control boxes,
    dynamic states, and validity masks. It also provides mappings for category and state names to their
    respective IDs, and utility methods for aggregation and querying.

    Attributes:
        transform (np.ndarray | torch.Tensor): Batched homogeneous 4x4 transformations ('world_T_box').
            Shape: (B>=1, T=1, N>=0, 4, 4).
        extent (np.ndarray | torch.Tensor): Extent of each traffic control element.
            Shape: (B>=1, T=1, N>=0, 3).
        category (np.ndarray | torch.Tensor): Category IDs for each element.
            Shape: (B>=1, T=1, N>=0).
        tce_id (np.ndarray | torch.Tensor): Unique IDs for each traffic control element.
            Shape: (B>=1, T=1, N>=0).
        control_box (np.ndarray | torch.Tensor): Control box described by 4 homogeneous vertices.
            Shape: (B>=1, T=1, N>=0, 4, 4).
        state (np.ndarray | torch.Tensor): Dynamic state of each element.
            Shape: (B>=1, T>=1, N>=0).
        category_map (dict[str, int]): Maps category names to category IDs.
        state_map (dict[str, int]): Maps state names to state IDs.
        is_valid (np.ndarray): Validity mask for each element.
            Shape: (B>=1, T>=1, N>=0).
        is_visible (np.ndarray): Visibility mask for each element. Shape: (B>=1, T>=1, N>=0).

    Properties:
        shape (tuple): Shape of the validity mask.

    Methods:
        _create_zeros(b: int = 1, t: int = 1, n: int = 1) -> TrafficControlElements:
            Creates a zero-initialized TrafficControlElements instance.
        aggregated_time(time: list[TrafficControlElements], use_custom_batching: list = None) -> TrafficControlElements:
            Aggregates single-batch and single-time traffic control elements over time.
        squeeze_invalid() -> TrafficControlElements:
            Returns a clone with all items marked as invalid removed.
        get_traffic_light_state(tce: TrafficControlElements, tl_id: int, b: int = 0, t: int = 0) -> int:
            Gets the traffic light state for a given ID.

    Note:
        The class assumes that TCEs with the same track ID occur at the same index in the tce_id column.
    """

    # static
    transform: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 1, 4, 4), dtype=np.float64)) Batched homogeneous 4x4 transformations 'world_T_box'. Shape: (B>=1, T=1, N>=0, 4, 4)
    extent: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 1, 3), dtype=np.float32)) Shape: (B>=1, T=1, N>=0)
    category: (
        np.ndarray | torch.Tensor
    )  # = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.uint8)) Shape: (B>=1, T=1, N>=0)
    tce_id: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.int32)) Shape: (B>=1, T=1, N>=0)
    control_box: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 1, 4, 4), dtype=np.float32)) Describes the control box by 4 homogenous vertices Shape: (B>=1, T=1, N>=0, 4, 4).  The vertices describe the box on the road surface as a polygon.
    # dynamic
    state: (
        np.ndarray | torch.Tensor | None
    )  # = field(default_factory=lambda: np.zeros((1, 1, 1), dtype=np.uint32)) Shape: (B>=1, T>=1, N>=0)
    is_visible: np.ndarray | torch.Tensor | None
    # label spec
    category_map: (
        dict[str, int] | None
    )  # = field(default_factory=dict) Maps category names (str) to a category ids (int).
    state_map: dict[str, int] | None  # = field(default_factory=dict) Maps state names (str) to a state ids (int).

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
        self._check_array_dim("category", 3, None, ignore_list)
        self._check_array_dim("tce_id", 3, None, ignore_list)
        self._check_array_dim("control_box", 2, (4, 4), ignore_list)
        self._check_array_dim("state", 3, None, ignore_list)
        self._check_array_dim("is_visible", 3, None, ignore_list)

    @classmethod
    def _create_zeros(cls: Type[TCE], b: int = 1, t: int = 1, n: int = 1) -> TCE:
        """
        Creates a new instance of the class with all fields initialized to zero arrays.
        This class method generates an object with all numerical fields set to arrays of zeros,
        and all mapping fields set to empty dictionaries. The shapes and data types of the arrays
        are determined by the input parameters.

        Args:
            b (int, optional): Batch size dimension. Defaults to 1.
            t (int, optional): Time steps dimension. Defaults to 1.
            n (int, optional): Number of elements per batch and time step. Defaults to 1.

        Returns:
            TCE: An instance of the class with all fields initialized to zeros or empty mappings.
        """

        return cls(
            is_valid=np.zeros((b, t, n), dtype=np.uint8),
            transform=np.zeros((b, t, n, 4, 4), dtype=np.float64),
            extent=np.zeros((b, t, n, 3), dtype=np.float32),
            category=np.zeros((b, t, n), dtype=np.uint8),
            tce_id=np.zeros((b, t, n), dtype=np.int32),
            control_box=np.zeros((b, t, n, 4, 4), dtype=np.float32),
            state=np.zeros((b, t, n), dtype=np.uint32),
            is_visible=np.ones((b, t, n), dtype=bool),
            category_map={},
            state_map={},
        )

    def transform_to(self, trans_matrix: np.ndarray | torch.Tensor) -> TrafficControlElements:
        """
        Transforms the traffic control elements to a given reference frame.

        Args:
            trans_matrix: The (4, 4) transform matrix to apply.

        Returns:
            transformed_state: A new TrafficControlElements object transformed to the given
            reference frame.
        """
        assert self.shape[0] == 1, "For now transforming batched data is not supported! "
        assert self.transform is not None
        new_transform = trans_matrix @ self.transform
        new_control_box = self.control_box @ np.transpose(trans_matrix[0])
        return TrafficControlElements(
            is_valid=self.is_valid,
            transform=new_transform,
            control_box=new_control_box,
            extent=self.extent,
            category=self.category,
            tce_id=self.tce_id,
            state=self.state,
            is_visible=self.is_visible,
            category_map=self.category_map,
            state_map=self.state_map,
        )

    @classmethod
    def aggregated_time(cls: Type[TCE], tces: List[TCE], use_custom_batching: list | None = None) -> TCE:
        """Aggregates single-batch and single-time traffic control elements (TCEs) over time.
        This method combines a list of TCE objects, each representing a single time step, into a single
        TrafficControlElements object that aggregates all unique TCEs across the provided time steps.
        TCEs with the same track id are aligned at the same index in the resulting structure.

        Args:
            tces (list[TCE]): A list of TCE objects, each corresponding to a different time step.
            use_custom_batching: to be compatible with base class "aggregated_time". Not used right now.

        Returns:
            TCE: A TrafficControlElements object containing the aggregated TCEs over all provided time steps.


        Notes:
            - Assumes that each input TCE contains tce data in the first batch and time index.
        """
        assert use_custom_batching is None, (
            "use_custom_batching is only included to properly overwrite base class. Do not set != None"
        )

        if not np.all([tce.is_valid is not None for tce in tces]):
            if np.any([tce.is_valid is not None for tce in tces]):
                raise ValueError("Mixing empty TCE with is_valid==None with ones with valid data is not supported.")
            return cls._create_zeros(b=1, t=len(tces), n=0)
        if not np.all([tce.tce_id is not None for tce in tces]):
            raise ValueError("aggregated_time() currently requires tce_id.")

        ftce_ids = [tce.tce_id[0, 0] for tce in tces]  # type: ignore[index]  # mypy doesn't transfer the not-None assertion above to here
        tce_ids = np.unique(np.concatenate(ftce_ids, axis=0).flatten()).astype(int)
        tce_ids.sort()
        tce_id_look_up = {}
        for i, fid in enumerate(tce_ids):
            tce_id_look_up[fid] = i

        res = cls._create_zeros(t=len(tces), n=len(tce_ids))

        for timestep, tce in enumerate(tces):
            assert tce.tce_id is not None  # this is asserted above already, mypy doesn't understand
            for tce_idx in range(tce.tce_id.shape[2]):
                tce_id = tce.tce_id[0, 0, tce_idx]

                idx = tce_id_look_up[tce_id]

                if res.transform is not None and tce.transform is not None:
                    res.transform[0, timestep, idx] = tce.transform[0, 0, tce_idx]
                if res.extent is not None and tce.extent is not None:
                    res.extent[0, timestep, idx] = tce.extent[0, 0, tce_idx]
                if res.category is not None and tce.category is not None:
                    res.category[0, timestep, idx] = tce.category[0, 0, tce_idx]
                if res.tce_id is not None and tce.tce_id is not None:
                    res.tce_id[0, timestep, idx] = tce_id
                if res.control_box is not None and tce.control_box is not None:
                    res.control_box[0, timestep, idx] = tce.control_box[0, 0, tce_idx]
                if res.state is not None and tce.state is not None:
                    res.state[0, timestep, idx] = tce.state[0, 0, tce_idx]

                res.is_valid[0, timestep, idx] = tce.is_valid[0, 0, tce_idx]
                if res.is_visible is not None and tce.is_visible is not None:
                    res.is_visible[0, timestep, idx] = tce.is_visible[0, 0, tce_idx]

        res.category_map = tces[0].category_map
        res.state_map = tces[0].state_map

        return res

    def squeeze_invalid(self) -> TrafficControlElements:
        """
        Returns a clone of this TCE instance with all items marked as invalid removed.
        This operation is only defined for instances where batch and time dimensions are equal to 1.
        Items are considered valid if their corresponding entry in `self.is_valid[0, 0]` equals 1.

        Returns:
            TrafficControlElements: A new instance containing only valid items.

        Raises:
            AssertionError: If batch dimension is not 1 or time dimension is not 1.
        """
        assert self.shape[0] == 1, f"op is not defined for batch dimension={self.shape[0]}"
        assert self.shape[1] == 1, f"op is not defined for time dimension={self.shape[1]}"

        valid_mask = self.is_valid[0, 0] == 1
        return self[0:1, 0:1, valid_mask]

    @classmethod
    def aggregate_tces(cls: Type[TCE], tces: list[TrafficControlElements]) -> TrafficControlElements:
        res = TrafficControlElements.create_empty()

        class_arrays = [
            k for k, v in tces[0].__dict__.items() if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)
        ]

        for k in class_arrays:
            res.__dict__[k] = cls._concat_tensors([tce.__dict__[k] for tce in tces], dim=2)

        res.category_map = tces[0].category_map
        res.state_map = tces[0].state_map

        return res

    @classmethod
    def get_traffic_light_state(cls: Type[TCE], tce: TrafficControlElements, tl_id: int, b: int = 0, t: int = 0) -> int:
        """Get traffic light state from TrafficControlElements for given ID

        Args:
            tce: TrafficControlElements object
            tl_id: traffic light ID to look up
            b: batch index
            t: time index

        Returns:
            TrafficLightState value
        """
        assert tce.tce_id is not None
        assert tce.state is not None

        # Find index where tce.id matches tl_id
        matches = (tce.tce_id[b, 0] == tl_id).nonzero()[0]
        if len(matches) == 0:
            return TrafficLightState.UNKNOWN

        tce_idx = matches[0]
        if not tce.is_valid[b, t, tce_idx]:
            return TrafficLightState.UNKNOWN

        return int(tce.state[b, t, tce_idx].item())


class TrafficControlElementsUtils:
    """Utility class for operations on TrafficControlElements.
    This class provides static methods to manipulate and transform
    TrafficControlElements objects, such as applying transformation
    matrices to their attributes.

    Methods:
        transform_traffic_control_elements: Applies a transformation matrix
            to the 'transform' and 'control_box' fields of a TrafficControlElements
            instance, returning a new instance with updated values.
    """

    @staticmethod
    def transform_traffic_control_elements(tce: TrafficControlElements, M: np.ndarray) -> TrafficControlElements:
        # TODO: Deprecated, use class member
        """Applies a transformation matrix to the `transform` and `control_box` fields of a TrafficControlElements object.
        The transformation matrix `M` is broadcasted and left-multiplied to the `transform` and `control_box` arrays
        of the input `tce`. All other fields are copied unchanged.

        Args:
            tce (TrafficControlElements): The input object containing traffic control elements with fields to be transformed.
            M (np.ndarray): A (4, 4) transformation matrix to apply.

        Returns:
            TrafficControlElements: A new TrafficControlElements object with updated `transform` and `control_box` fields.

        Raises:
            ValueError: If the shapes of `M`, `tce.transform`, or `tce.control_box` are incompatible for broadcasting.

        Note:
            - `M` should have shape (4, 4).
            - `tce.transform` and `tce.control_box` should have shape (B, T, N, 4, 4).
        """
        # Expand M to (1,1,1,4,4) so it can broadcast with (B,T,N,4,4)
        new_transform = M @ tce.transform
        new_control_box = M @ tce.control_box

        return TrafficControlElements(
            transform=new_transform,
            control_box=new_control_box,
            extent=tce.extent,
            category=tce.category,
            tce_id=tce.tce_id,
            state=tce.state,
            is_visible=tce.is_visible,
            is_valid=tce.is_valid,
            category_map=tce.category_map,
            state_map=tce.state_map,
        )
