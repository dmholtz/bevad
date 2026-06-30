from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
from typing_extensions import Self


@dataclass
class BaseEntity(ABC):
    """Abstract base class representing a base entity with indexing and aggregation capabilities.

    This class provides foundational functionality for entities that require batch and time
    dimension handling, tensor operations, and data validation. It serves as a framework
    for implementing complex data structures with indexing, aggregation, and shape
    management capabilities.

    The class is designed to work with multi-dimensional data where the first dimension
    typically represents batch size (B) and the second dimension represents time (T).

    Key features:
    - Index normalization and validation for safe data access
    - Tensor concatenation utilities for combining data
    - Empty instance creation for initialization and fill operations
    - Abstract shape definition requiring subclass implementation
    - Batch and time aggregation methods for data combination
    - Validity tracking through boolean arrays

    Attributes:
        is_valid: Optional numpy array of booleans indicating the validity of entity
            data. Expected shape is (B, T, ...) where B is batch size, T is time
            steps, and additional dimensions depend on the specific entity type.

    Note:
        Subclasses must implement all abstract methods and properties to provide
        concrete functionality for specific entity types.
    """

    is_valid: np.ndarray | torch.Tensor  # array of booleans indicating validity of entity data (B, T, ...)

    def _check_array_dim(
        self, array_name, common_dimensions: int, custom_dimensions: Tuple | None, ignore_list: List[str] | None = None
    ):
        """Check if array dimension match the target dimensions.
        Args:
            array_name: The name of the array to check
            common_dimensions: The number of common dimensions with the is_valid tensor. I.e. batch, time and
                                    elements. For some data classes it may be only batch or batch and time.
            custom_dimensions: The dimensionality of the actual data e.g. 4x4 for a transformation matrix.
            ignore_list: class elements to ignore for checking.
        Note:
             Will raise an assert if dimensions do not match

        """
        assert array_name in self.__dict__, "Error: " + array_name + " not found in " + type(self).__name__
        if ignore_list is not None and type(self).__name__ + "." + array_name in ignore_list:
            return

        array = self.__dict__[array_name]
        iv_shape = self.is_valid.shape
        if array is not None:
            arr_shape = array.shape
            assert arr_shape[:common_dimensions] == iv_shape[:common_dimensions]
            if custom_dimensions is not None:
                assert arr_shape[-len(custom_dimensions) :] == custom_dimensions, "Invalid array dimensions!"

    def _check_list_dim_rec(self, is_valid, lists: Any, common_dimensions: int, shape_index: int):
        """Check if list dimensions match the is_valid dimensions. Recursive verion.
        Args:
            is_valid: The is_valid array, indicating of an element is valid.
            lists: The lists to check
            common_dimensions: The number of common dimensions with the is_valid tensor. I.e. batch, time and
                                    elements. For some data classes it may be only batch or batch and time.
            shape_index: The shape index, i.e. the index of the shape value to check.
        Note:
             ill raise an assert if dimensions do not match

        """
        ### TODO: check for custom list dimensions
        if shape_index >= common_dimensions:
            return
        assert is_valid is not None, "is_valid is None!"
        if len(is_valid.shape) == 1:
            # assert np.where(is_valid == True)[0].shape[0] == len(lists), "Invalid list dimensions!"
            assert lists is not None, "Lists is None!"
            assert is_valid.shape[0] == len(lists), "Invalid list dimensions!"
        else:
            assert self.is_valid.shape[0] == len(lists)
            if lists is not None:
                for i, l in enumerate(lists):
                    if l is not None and isinstance(l, List):
                        self._check_list_dim_rec(self.is_valid[i], l, common_dimensions, shape_index + 1)

    def _check_list_dim(
        self,
        lists_name: Any,
        common_dimensions: int,
        custom_dimensions: Tuple | None,
        ignore_list: List[str] | None = None,
    ):
        """Check if list dimensions match the is_valid dimensions.
        Args:
            lists_name: The name of the lists to check
            common_dimensions: The number of common dimensions with the is_valid tensor. I.e. batch, time and
                                    elements. For some data classes it may be only batch or batch and time.
            custom_dimensions: The dimensionality of the actual data e.g. 4x4 for a transformation matrix.
            ignore_list: class elements to ignore for checking.
        Note:
             ill raise an assert if dimensions do not match

        """
        assert lists_name in self.__dict__, "Error: " + lists_name + " not found in " + type(self).__name__
        if ignore_list is not None and type(self).__name__ + "." + lists_name in ignore_list:
            return
        lists = self.__dict__[lists_name]

        self._check_list_dim_rec(self.is_valid, lists, common_dimensions, 0)

    @abstractmethod
    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        """Actual implementation of the data dimension check. Each derived class needs to override this and call the
        check for each member with target dimension.
        """
        pass

    def check_data_dimensions(self, ignore_list: List[str] | None = None):
        """Will call _check_data_dimensions_impl to verify the correct dimensionality of the data.
        Will test it for the current class and all members of type base entity.
        """
        self._check_data_dimensions_impl(ignore_list)
        for be in self._filter_base_entity().values():
            be.check_data_dimensions(ignore_list)

    @staticmethod
    def _get_fill_value(obj: torch.Tensor | np.ndarray) -> Any:
        """Determines an appropriate fill value based on the type and dtype of the input.

        This method returns a suitable default fill value for the provided object,
        depending on whether it's a PyTorch tensor or a NumPy array, and based on its data type:

        - Floating point types: NaN
        - Integer types: 0
        - Boolean types: False
        - Other types: None

        Args:
            obj (torch.Tensor | np.ndarray): A PyTorch tensor or NumPy array
                (or an object with a `.dtype` attribute) for which a fill value
                should be determined.

        Returns:
            Any: The default fill value appropriate for the input's data type.
                Returns `None` if the data type is unrecognized or unsupported.

        Note:
            For objects without a `dtype` attribute, the function attempts to infer a
            NumPy dtype. If that fails, it returns `None`.
        """

        # Handle PyTorch tensors
        if isinstance(obj, torch.Tensor):
            if torch.is_floating_point(obj):
                return float("nan")
            elif obj.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                return 0
            elif obj.dtype == torch.bool:
                return False
            else:
                return None

        # Try to extract a NumPy dtype safely
        try:
            dtype = obj.dtype if hasattr(obj, "dtype") else np.dtype(obj)
        except TypeError:
            return None  # or raise error if needed

        # Handle NumPy types
        if np.issubdtype(dtype, np.floating):
            return np.nan
        elif np.issubdtype(dtype, np.integer):
            return 0
        elif np.issubdtype(dtype, np.bool_):
            return False
        else:
            return None

    def _filter_base_entity(self):
        """Filters and returns attributes that are instances of BaseEntity.

        Iterates through the instance's `__dict__` and collects all key-value pairs
        where the value is an instance of `BaseEntity`.

        Returns:
            dict: A dictionary containing attribute names and values where each value
            is an instance of `BaseEntity`.
        """

        return {k: v for k, v in self.__dict__.items() if isinstance(v, BaseEntity)}

    def shallow_copy(self) -> Self:
        """Creates a shallow copy of a base entity.
        Returns:
            A new instance of same type with all members shallow copied."""
        res = self.__class__.create_empty()
        for k, v in self.__dict__.items():
            res.__dict__[k] = v
        return res

    def to_dict(self) -> dict:
        """
        Creates a dict representation of a base entity.
        Returns:
            A dict with all members converted to dict.
        """
        res = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BaseEntity):
                res[k] = v.to_dict()
            else:
                res[k] = v
        return res

    def _to_list_idx(self, idx: tuple) -> tuple:
        """Converts an indexing tuple to a form suitable for slicing, handling ellipsis.

        Iterates over elements in the input `idx` tuple and replaces any Python ellipsis
        (`...`) with a full slice (`slice(None, None, None)`). Processing stops at the
        first ellipsis encountered.

        Args:
            idx (tuple): A tuple of indices, which may include an ellipsis.

        Returns:
            tuple: A modified tuple with ellipsis replaced by a full slice, suitable
            for list or array indexing.
        """

        res = []

        for i, v in enumerate(idx):
            if v.__class__.__name__ == "ellipsis":
                res.append(slice(None, None, None))
                break
            else:
                res.append(v)
        return tuple(res)

    def _normalize_idx(self, idx: int | tuple, n: int) -> tuple:
        """Normalizes indexing input into a tuple of slices.

        This method ensures the indexing tuple is valid and transforms all integer
        indices into equivalent slice objects. It also enforces that ellipsis (`...`)
        is only allowed as the last element of the index tuple.

        Args:
            idx (tuple): A tuple containing integer indices, slices, or an ellipsis.
            n (int): Expected number of dimensions (not directly used, but may be
                useful for validation or expansion in future versions).

        Returns:
            tuple: A tuple where all integer elements have been converted to
            `slice(i, i+1)` and ellipsis has been validated.

        Raises:
            ValueError: If the ellipsis appears in any position other than the end,
            or if `idx` is not a tuple.

        Example:
            >>> self._normalize_idx((2, ...), n=3)
            (slice(2, 3, 1), Ellipsis)
        """

        if isinstance(idx, int):
            idx = (idx,)

        for i, v in enumerate(idx):
            if v.__class__.__name__ == "ellipsis" and i < len(idx) - 1:
                raise ValueError(
                    f"_normalize_idx: ellipsis is only allowed as last element, but is at {i} for idx of "
                    f"length {len(idx)}"
                )

        if not isinstance(idx, tuple):
            raise ValueError(f"_normalize_idx is not defined for idx of type {type(idx)}")

        def int_to_slice(int_val):
            return slice(int_val, int_val + 1, 1)

        # Normalize index
        idx = tuple(int_to_slice(i) if isinstance(i, int) else i for i in idx)

        return idx

    def _slice_list(self, idx, l: list[Any]) -> list[Any]:
        """Slices a nested list along batch and time dimensions.

        This method assumes `l` is a list of lists, where the outer list represents
        the batch dimension and the inner lists (if present) represent the time dimension.
        It applies the given index tuple `idx` to slice both dimensions accordingly.

        This method can be overridden in derived classes to implement custom slicing logic,
        particularly when the list structure differs from the default assumption.

        Args:
            idx: A tuple of indices or slices (batch_idx, time_idx).
            l (list[Any]): A nested list representing [batch][time], or a flat list representing [batch].
                If `l` is `None`, no slicing is performed.

        Returns:
            list[Any]: A sliced subset of the input list `l`, respecting the specified indices.

        Example:
            >>> l = [[1, 2], [3, 4], [5, 6]]
            >>> self._slice_list((slice(1, 3), slice(0, 1)), l)
            [[3], [5]]
        """
        res: list[Any] = []

        for b in l[idx[0]]:
            ### if b is another list, there is a time dimension
            if isinstance(b, list) and len(idx) > 1:
                res.append([])
                for t in b[idx[1]]:
                    res[-1].append(t)
            else:
                ### if b is something else, by default assume there is no separate time dimension
                res.append(b)

        return res

    @classmethod
    def _concat_tensors(
        cls,
        tensors: list[np.ndarray | torch.Tensor],
        dim: int,
        max_dim: list[int] | None = None,
        fill_up_to_max: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Concatenates a list of tensors along a given dimension.

        Dispatches to the appropriate backend (NumPy or PyTorch) depending on the type
        of the input tensors. Supports optional dimension capping and padding.

        Args:
            tensors (list[np.ndarray | torch.Tensor]): List of tensors to concatenate.
            dim (int): The dimension along which to concatenate.
            max_dim (list[int] | None): If provided, each dimension X will be capped at max_dim[X]
                (if > 0). Values of 0 mean no limit. If None, defaults to zeros (no capping).
            fill_up_to_max (bool): If True, pads tensors with a default fill value up to `max_dim`.

        Returns:
            np.ndarray | torch.Tensor: The concatenated and optionally padded tensor.

        Raises:
            NotImplementedError: If the tensor type is not supported.
        """

        if isinstance(tensors[0], np.ndarray):
            numpy_tensors = cast(List[np.ndarray], tensors)
            return cls._concat_tensors_np(numpy_tensors, dim, max_dim, fill_up_to_max)

        if isinstance(tensors[0], torch.Tensor):
            torch_tensors = cast(List[torch.Tensor], tensors)
            return cls._concat_tensors_torch(torch_tensors, dim, max_dim, fill_up_to_max)

        raise NotImplementedError()

    @classmethod
    def _concat_tensors_np(
        cls, tensors: list[np.ndarray], dim: int, max_dim: list[int] | None, fill_up_to_max
    ) -> np.ndarray:
        """Concatenates a list of NumPy arrays along a specified dimension with optional padding.

        Pads arrays to uniform shape before concatenation if needed. Optionally fills to a fixed
        `max_dim` shape using a type-appropriate fill value (e.g., 0, NaN).

        Args:
            tensors (list[np.ndarray]): List of NumPy arrays to concatenate.
            dim (int): Axis along which to concatenate the arrays.
            max_dim (list[int] | None): Maximum allowed size for each dimension. A value of 0 means
                no restriction for that dimension. If None, it's initialized to 0s.
            fill_up_to_max (bool): If True, pads arrays with a fill value to match `max_dim`.

        Returns:
            np.ndarray: A single array created by stacking the inputs along `dim`.

        """
        # if all shapes are equal, we can directly concatenate it
        test_shape = tensors[0].shape
        all_shapes_equal = True
        for t in tensors:
            if t.shape != test_shape:
                all_shapes_equal = False
                break

        if all_shapes_equal:
            return np.concatenate([t for t in tensors], axis=dim)

        shapes = np.array([t.shape for t in tensors])
        if max_dim is None:
            max_dim_arr = np.zeros_like(tensors[0].shape)
        else:
            max_dim_arr = np.array(max_dim)
        # get max from tensors
        max_shape = np.max(shapes, axis=0)

        # if no max is specified use max from tensors
        max_dim_arr[max_dim_arr == 0] = max_shape[max_dim_arr == 0]

        # if filling is requested create data with size max_dim
        if fill_up_to_max:
            new_dim = max_dim_arr
        else:
            # else clip to maximum from tensors
            new_dim = np.clip(max_dim_arr, [0] * len(max_dim_arr), [max_shape])

        new_dim = new_dim.flatten()
        # set the stack dimension to number of tensors
        new_dim[dim] = len(tensors)
        fill_value = cls._get_fill_value(tensors[0])
        res = np.full(new_dim, fill_value=fill_value)

        for i, t in enumerate(tensors):
            # get minimum from src and dst shape
            t_shape = np.min(np.stack((np.array(t.shape), np.array(new_dim)), axis=1), axis=1)
            src_idx = [slice(0, s, None) for s in t_shape[:dim]] + [0] + [slice(0, s, None) for s in t_shape[dim + 1 :]]
            dst_idx = [slice(0, s, None) for s in t_shape[:dim]] + [i] + [slice(0, s, None) for s in t_shape[dim + 1 :]]
            res[tuple(dst_idx)] = t[tuple(src_idx)]

        return res

    # TODO : need to check this
    @classmethod
    def _concat_tensors_torch(
        cls, tensors: list[torch.Tensor], dim: int, max_dim: list[int] | None, fill_up_to_max
    ) -> torch.Tensor:
        """Concatenates a list of PyTorch tensors along a specified dimension with optional padding.

        Handles shape mismatches by padding to uniform shape before stacking. Pads tensors up to
        `max_dim` if requested, using a suitable fill value (e.g., 0 for integers, NaN for floats).

        Args:
            tensors (list[torch.Tensor]): List of PyTorch tensors to concatenate.
            dim (int): Axis along which to concatenate.
            max_dim (list[int] | None): Maximum shape constraints. A value of 0 implies no restriction.
            fill_up_to_max (bool): If True, pads tensors to match `max_dim`.

        Returns:
            torch.Tensor: A single tensor created by concatenating the inputs along `dim`.

        """
        # if all shapes are equal, we can directly concatenate it
        test_shape = tensors[0].shape
        all_shapes_equal = True
        for t in tensors:
            if t.shape != test_shape:
                all_shapes_equal = False
                break

        if all_shapes_equal:
            return torch.cat([t for t in tensors], dim=dim)

        shapes = np.array([t.shape for t in tensors])

        # TODO : need to check if max_dim is None ?
        #        need to check if max_dim has same len as max_shape ?

        max_dim_arr = np.array(max_dim)
        # get max from tensors
        max_shape = np.max(shapes, axis=0)

        # if no max ist specified use max from tensors

        max_dim_arr[max_dim_arr == 0] = max_shape[max_dim_arr == 0]

        # if filling is requested create data with size max_dim
        if fill_up_to_max:
            new_dim = max_dim_arr
        else:
            # else clip to maximum from tensors
            new_dim = np.clip(max_dim_arr, [0] * len(max_dim_arr), [max_shape])

        new_dim = new_dim.flatten()
        # set the stack dimension to number of tensors
        new_dim[dim] = len(tensors)

        fill_value = cls._get_fill_value(tensors[0])
        res = torch.full(new_dim.tolist(), fill_value, dtype=tensors[0].dtype, device=tensors[0].device)

        for i, t in enumerate(tensors):
            # get minimum from src and dst shape
            t_shape = np.min(np.stack((np.array(t.shape), np.array(new_dim)), axis=1), axis=1)
            src_idx = [slice(0, s, None) for s in t_shape[:dim]] + [0] + [slice(0, s, None) for s in t_shape[dim + 1 :]]
            dst_idx = [slice(0, s, None) for s in t_shape[:dim]] + [i] + [slice(0, s, None) for s in t_shape[dim + 1 :]]
            res[tuple(dst_idx)] = t[tuple(src_idx)]

        return res

    def __getitem__(self, idx: int | Tuple, use_custom_slicing: List[str] | None = None) -> Self:
        """Retrieve a subset or slice of the entity according to the specified index.

        Supports slicing across multiple internal attributes, including nested BaseEntity
        instances, NumPy arrays, PyTorch tensors, lists, and dictionaries. Allows skipping
        default slicing for specified members to enable custom slicing behavior.

        Args:
            idx (int, tuple): Index or slice specification to select elements.
            use_custom_slicing (list | None): Optional list of attribute names for which
                the default slicing behavior is disabled.

        Returns:
            Self: A new instance of the entity class containing the selected subset of data.
        """
        if use_custom_slicing is None:
            use_custom_slicing = []

        idx = self._normalize_idx(idx, 2)

        res = self.__class__.create_empty()
        for k, v in self.__dict__.items():
            if k in use_custom_slicing:
                continue

            if isinstance(v, BaseEntity):
                res.__dict__[k] = v[idx]

            if isinstance(v, np.ndarray):
                ### Limit to the number of array dimensions. TODO: Check if there are issues with that
                tidx = idx[: len(v.shape)]
                res.__dict__[k] = v[tidx]

            if isinstance(v, torch.Tensor):
                ### Limit to the number of array dimensions. TODO: Check if there are issues with that
                tidx = idx[: len(v.shape)]
                res.__dict__[k] = v[tidx]

            # dicts are not slice-able by default. If a dict consists of slice-able members, implement a custom slicing
            if isinstance(v, dict):
                res.__dict__[k] = v

            ### This assumes first list is batch dimension
            if isinstance(v, list):
                res.__dict__[k] = self._slice_list(idx, v)

        # cc.__dict__.update({k: v[idx] for k, v in self._filter_base_entity().items()})

        return res

    @classmethod
    def aggregated_batch(
        cls, batch: list[Self], use_custom_batching: list | None = None, max_t: int = 0, fill_up_to_max: bool = False
    ) -> Self:
        """Aggregate a list of entity instances into a single batched entity.
        The aggregation process:
            - NumPy arrays: Concatenated along axis 0 with optional padding
            - PyTorch tensors: Concatenated along axis 0 with optional padding
            - Nested BaseEntity instances: Recursively batched
            - Lists: First element taken from each batch item
            - Other types: Value from first batch item is used

        Args:
            batch (List[BaseEntity]): List of entity instances to aggregate.
                All instances must be of the same type and have compatible shapes.
            use_custom_batching (Optional[List[str]]): List of attribute names that
                should be skipped during automatic batching.
            max_t (int): Maximum time dimension size. If > 0, tensors will be
                padded/truncated to this size along dimension 1 (time axis).
                If 0, uses the maximum time dimension found in the batch.
            fill_up_to_max (bool): Whether to pad tensors to max dimensions.
                If False, tensors are concatenated as-is. If True, smaller tensors
                are padded to match the largest dimensions.

        Returns:
            BaseEntity: A new entity instance with all compatible attributes
                batched along dimension 0. The batch size will equal len(batch).

        """

        ### TODO: Check if max_t is required. What is the use case for not just using max-of tensor shapes?
        if use_custom_batching is None:
            use_custom_batching = []

        res = cls.create_empty()

        for k, v in batch[0].__dict__.items():
            if k in use_custom_batching:
                continue

            if isinstance(v, BaseEntity):
                res.__dict__[k] = v.aggregated_batch([b.__dict__[k] for b in batch])
                continue

            if isinstance(v, np.ndarray):
                max_dim = [0] * len(v.shape)
                if max_t > 0:
                    max_dim[1] = max_t
                res.__dict__[k] = cls._concat_tensors_np(
                    [b.__dict__[k] for b in batch], 0, max_dim=max_dim, fill_up_to_max=fill_up_to_max
                )
                continue
                # res.__dict__[k] = np.concatenate([b.__dict__[k] for b in batch], axis=0)

            if isinstance(v, torch.Tensor):
                max_dim = [0] * len(v.shape)
                if max_t > 0:
                    max_dim[1] = max_t

                res.__dict__[k] = cls._concat_tensors_torch(
                    [b.__dict__[k] for b in batch], 0, max_dim=max_dim, fill_up_to_max=fill_up_to_max
                )
                continue
                # res.__dict__[k] = torch.concatenate([b.__dict__[k] for b in batch], axis=0)

            if isinstance(v, list):
                # TODO (huuangu): batching two values of shape (N, ) and (M, ) might not be a good idea because the final shape (2, max(N, M))
                # would have dimensionality of 2, which differs from the original dimensionality of 1. I.e. we will have core container field
                # with multiple dimension lengths
                # Maybe should we have an originial shape (1, M) and (1, N) instead? -> discussion
                if batch[0].dimensionality > 1:
                    ### TODO: this assumes the second list dimension is time. Is this always correct?
                    if isinstance(v[0], List):
                        max_dim_scalar = max([len(b.__dict__[k][0]) for b in batch])
                        res.__dict__[k] = [
                            b.__dict__[k][0] + [None] * (max_dim_scalar - len(b.__dict__[k][0])) for b in batch
                        ]
                    else:
                        # Not optimal yet -> we should have an original consistent dimension length of 2 (e.g. (1, N)) instead of (N,)
                        unique = set()
                        for b in batch:
                            unique |= set(b.__dict__[k])
                        res.__dict__[k] = list(unique)

                else:
                    res.__dict__[k] = [b.__dict__[k][0] for b in batch]
                continue

            ### for non-batchable types just use value from first batch entry
            # print("Non-Batch-Batchable: " + k)
            res.__dict__[k] = v

            # # dicts are not batch-able by default. If a dict consists of batch-able members, implement a custom batching
            # if isinstance(v, dict):
            #     res.__dict__[k] = v
            #
            # # strings are not batch-able by default. If a strings should be batched, implement a custom batching
            # if isinstance(v, str):
            #     res.__dict__[k] = v

        # for k, v in batch[0]._filter_base_entity().items():
        #    res.__dict__[k] = v.aggregate_batch([e._filter_base_entity()[k] for e in batch])
        return res

    @classmethod
    def aggregated_time(cls, batch: list[Self], use_custom_batching: list | None = None) -> Self:
        """Aggregate a batch of instances along the time dimension.

        The aggregation process:
            - BaseEntity objects: Recursively aggregated using their own aggregated_time method
            - NumPy arrays/Torch tensors: Concatenated along axis/dimension 1 (time axis)
            - Lists: Combined by taking the first element from each batch item
            - Other types: Use value from the first batch entry (non-batchable)

        Args:
            batch: List of class instances to aggregate. All instances should have
                compatible data structures and shapes for time dimension concatenation.
                None values in the batch are handled by using fill values.
            use_custom_batching: Optional list of attribute names that should be
                skipped during standard aggregation.
        Returns:
            A new instance of the class containing the time-aggregated data from all
            batch items.

        TODO:
            - do we need checks for b==1 and t==1? I guess the user should just take care ?
            - Do we need to handle if something is "None" for batch[0] but has a value otherwise?
        """

        if use_custom_batching is None:
            use_custom_batching = []

        ### TODO: do we need checks for b==1 and t==1? I guess the user should just take care...
        ### I guess ideally it should also work for b>1 and t=1

        res = cls.create_empty()
        ref_item = batch[0]
        for i in range(len(batch)):
            if ref_item is not None:
                break
            ref_item = batch[i]

        for k, v in ref_item.__dict__.items():
            if k in use_custom_batching:
                continue

            ### TODO: Do we need to handle if something is "None" for batch[0] but has a value otherwise?
            ### Does this handle it correctly?
            if v is None:
                # Optionally check other items in batch for non-None values
                for i in range(1, len(batch)):
                    alt_v = getattr(batch[i], k, None)
                    if alt_v is not None:
                        v = alt_v  # Use this value instead
                        break

            if isinstance(v, BaseEntity):
                # Get one valid b to generate fill obj
                valid_b = next(b for b in batch if b is not None and k in b.__dict__)
                fill_value = valid_b.create_empty()
                res.__dict__[k] = v.aggregated_time([b.__dict__[k] if b is not None else fill_value for b in batch])
                continue

            if isinstance(v, np.ndarray):
                # Get one valid b to determine the shape
                valid_b = next(b for b in batch if b is not None and k in b.__dict__)
                shape = valid_b.__dict__[k].shape
                dtype = valid_b.__dict__[k].dtype

                fill_value = cls._get_fill_value(valid_b.__dict__[k])
                res.__dict__[k] = np.concatenate(
                    [b.__dict__[k] if b is not None else np.full(shape, fill_value, dtype=dtype) for b in batch],
                    axis=1,
                )
                continue

            if isinstance(v, torch.Tensor):
                # Get one valid b to determine the shape
                valid_b = next(b for b in batch if b is not None and k in b.__dict__)
                shape = valid_b.__dict__[k].shape
                dtype = valid_b.__dict__[k].dtype

                fill_value = cls._get_fill_value(valid_b.__dict__[k])

                res.__dict__[k] = torch.concatenate(
                    [
                        b.__dict__[k]
                        if b is not None
                        else torch.full(shape, fill_value, dtype=dtype, device=valid_b.__dict__[k].device)
                        for b in batch
                    ],
                    dim=1,
                )
                continue

            if isinstance(v, list):
                res.__dict__[k] = [[b.__dict__[k][0][0] if b is not None else None for b in batch]]
                continue

            ### for non-batchable types just use value from first batch entry
            # print("Non-Time-Batchable: " + k)
            res.__dict__[k] = v

            # dicts are not batch-able by default. If a dict consists of batch-able members, implement a custom batching
            # if isinstance(v, dict):
            #     res.__dict__[k] = v
            #
            # # strings are not batch-able by default. If a strings should be batched, implement a custom batching
            # if isinstance(v, str):
            #     res.__dict__[k] = v

        # for k, v in batch[0]._filter_base_entity().items():
        #    res.__dict__[k] = v.aggregate_batch([e._filter_base_entity()[k] for e in batch])
        return res

    def _has_ellipsis(self, idx):
        """Check if the indexing tuple contains an ellipsis (...).

        This method determines whether any element in the provided indexing tuple
        is an ellipsis object, which is used in NumPy-style advanced indexing to
        represent "all remaining dimensions".

        Args:
            idx (tuple or list): The indexing tuple/list to check. Each element
                can be a slice, integer, ellipsis, or other indexing object.

        Returns:
            bool: True if any element in idx is an ellipsis, False otherwise.

        """

        for i, v in enumerate(idx):
            if v.__class__.__name__ == "ellipsis":
                return True
        return False

    @property
    def shape(self) -> tuple:
        """Abstract property defining the entity's dimensional shape.

        Each BaseEntity subclass should at least have batch and time dimensions.

        Returns:
            Tuple[int, ...]: Shape of the entity as a tuple of integers. Must contain
            at least two dimensions (batch, time).
        """
        return self.is_valid.shape
        # pass

    @property
    @abstractmethod
    def dimensionality(self) -> int:
        """Abstract property defining the entity's dimensionality.

        Returns:
            Returns: int: Number of dimension = len(self.shape)
                        The first three dimensions MUST match: batch(required), time(optional), elements(optional).
                        Further dimensions can be custom.

        """
        pass

    @property
    def b_dim(self) -> int | None:
        """Returns the size of the batch dimension. There always is a batch dimension"""
        return self.is_valid.shape[0]

    @property
    @abstractmethod
    def t_dim(self) -> int | None:
        """Returns the size of the time dimension. Return none if object has no time dimension."""
        pass

    @property
    @abstractmethod
    def n_dim(self) -> int | None:
        """Returns the size of the element dimension. Return none if object has no element dimension."""
        pass

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty instance with all dataclass fields set to None.

        This factory method creates a new instance of the entity class with all
        dataclass fields initialized to None. This is useful for creating placeholder
        entities or initializing entities that will be populated later through
        attribute assignment or deserialization.

        Returns:
            BaseEntity: A new instance of the entity with all fields set to None.
        """

        # TODO: we cannot fully solve the wrong creation of class instances here.
        return cls(**{k: None for k in cls.__dataclass_fields__.keys()})  # type: ignore


class BaseEntityPayloadMixin(ABC):
    """Mixin class providing data payload handling functionality.

    Requires implementing classes to handle data loading/saving
    and dictionary instantiation.
    """

    def load_data(self, base_path: str) -> None:
        pass

    def save_data(self, base_path: str) -> None:
        pass

    @classmethod
    def create_from_dict(cls, obj_dict: dict):
        pass


class BaseEntityUtils:
    @staticmethod
    def _compare_recursive(v1: Any, v2: Any, key_path: str, ignore_list: List):
        """Test of two objects are equal with respect to the contained data.
        Also checks recursively for all member fields.

                    Args:
                        v1: The first object.
                        v2: The second object
                        key_path: the "name" of the object. Relevant for the ignore list.
                        ignore_list: the fields to ignore. Members are separated by ".".
                    Returns:
                        True if both objects are equal.

        """
        if key_path in ignore_list:
            return True

        if type(v1).__name__ != type(v2).__name__:
            print("Types are unequal: " + key_path)
            return False

        if isinstance(v1, BaseEntity):
            return BaseEntityUtils._compare_recursive(v1.__dict__, v2.__dict__, key_path, ignore_list=ignore_list)

        if isinstance(v1, List):
            if len(v1) != len(v2):
                print("List length differs: " + str(len(v1)) + " : " + str(len(v2)) + " Path: " + key_path)
                return False
            for i in range(len(v1)):
                are_equal = BaseEntityUtils._compare_recursive(v1[i], v2[i], key_path, ignore_list)
                if not are_equal:
                    return False
            return True

        if isinstance(v1, Dict):
            if len(v1) != len(v2):
                print("Dict length differs: " + str(len(v1)) + " : " + str(len(v2)) + " Path: " + key_path)
                return False
            diff1 = set(v1) - set(v2)
            if len(diff1) > 0:
                print("Dict keys differs: " + str(diff1) + " Path: " + key_path)
                return False
            for key in v1.keys():
                are_equal = BaseEntityUtils._compare_recursive(
                    v1[key], v2[key], key_path + str(".") + str(key), ignore_list
                )
                if not are_equal:
                    return False
            return True

        if isinstance(v1, np.ndarray):
            if v1.shape != v2.shape:
                print("Shapes are unequal: " + str(v1.shape) + str(" != ") + str(v2.shape) + key_path)
                return False
            if not np.allclose(v1, v2):
                print("Values are unequal: " + key_path)
                return False
            return True

        if isinstance(v1, torch.Tensor):
            if v1.shape != v2.shape:
                print("Shapes are unequal: " + str(v1.shape) + str(" != ") + str(v2.shape) + key_path)
                return False
            if not torch.allclose(v1, v2):
                print("Values are unequal: " + key_path)
                return False
            return True

        if "__dict__" in dir(v1):
            return BaseEntityUtils._compare_recursive(v1.__dict__, v2.__dict__, key_path, ignore_list=ignore_list)

        if v1 != v2:
            print("Values are unequal: " + key_path)
            return False

        ### Here we assume that the datatype implemented a proper == / != function. Handling all types individually
        # is not possible because of the custom fields in the container class that may contain other data
        return True

    @staticmethod
    def compare_entities(e1: BaseEntity, e2: BaseEntity, current_path: str = "cc", ignore_list: List | None = None):
        """Test of two base entities are equal. Also checks recursively for all member fields.

        Args:
            e1: The first base entity.
            e2: The second base entity
            current_path: the base "name" of the core container. Relevant for the ignore list.
            ignore_list: the fields to ignore. Members are separated by ".".
        Returns:
            True if both entities are equal.

        """

        if ignore_list is None:
            ignore_list = []

        return BaseEntityUtils._compare_recursive(e1.__dict__, e2.__dict__, current_path, ignore_list)
