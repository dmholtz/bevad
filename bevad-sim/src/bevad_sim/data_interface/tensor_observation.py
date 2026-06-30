from __future__ import annotations

import logging
import os
import pathlib
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np
import torch
from typing_extensions import Self

from bevad_sim.data_interface.base_entity import BaseEntity
from bevad_sim.data_interface.data_types import SensorTypes
from bevad_sim.data_interface.fragment_cache import fragment_cache

CO = TypeVar("CO", bound="CameraObservation")
LO = TypeVar("LO", bound="LidarObservation")
RO = TypeVar("RO", bound="RadarObservation")

logger = logging.getLogger(__name__)


@dataclass
class TensorObservation(BaseEntity):
    """Stores non-privileged, tensor-shaped observations from the world.
    This class represents a container for sensor observations, where each observation is stored as a tensor (NumPy array or PyTorch tensor).
    It supports batching along the batch, time, and sensor axes, and provides methods for loading, saving, and serializing observation data.

    Attributes:
        container_name (str): The name of this tensor observation container. Examples: 'rgb', 'depth', 'lidar', 'fisheye'.
        data (Optional[np.ndarray | torch.Tensor]): The data tensor of this sensor observation. Shape depends on the sensor type: (B>=1, T>=1, N>=1, ...).
        timestamps (np.ndarray | torch.Tensor): The timestamps of each sensor observation. Shape: (B>=1, T>=1, N>=1).
        frame_ids (np.ndarray | torch.Tensor): The integer frame ids of each sensor observation. Shape: (B>=1, T>=1, N>=1).
        sensor_names (list[str]): List of names for each sensor along the entity dimension (N). Shape: (N,).
        fileformat (str): The default file format for serializing a single observation in data, e.g., 'jpg' or 'bin'.

    Properties:
        b_dim (int): The batch dimension size.
        t_dim (int): The time dimension size.
        n_dim (int): The sensor dimension size.
        shape (tuple[int, int, int]): The shape of the observation (batch, time, sensor).

    Methods:
        build_sensor_batch(observations): Build a new TensorObservation from a list of TensorObservations by batching them along the sensor axis.
        get_idx_filename_pairs(base_path): Return ((t_idx, s_idx), filename) tuples for each time- and sensor-batched observation.
        in_memory(): Return True if the data associated with this observation is in memory (CPU/GPU).
        load_data(base_path): Load the data associated with this observation into CPU memory.
        save_data(base_path): Save the data payload of this tensor observation to disk.
        serialize_data_class(base_path): Serialize the data class, saving data and returning a dictionary representation.
        unload_data(): Delete the data associated with this observation from memory (CPU/GPU).
        _init_data(): Allocate blank tensors for loading the data payload into memory (abstract).
        _load_fragment(t_idx, s_idx, filename): Load a fragment of the data from disk (abstract).
        _save_fragment(t_idx, s_idx, filename): Save a fragment of the data to disk (abstract).
        get_sensor_type(): Return the type of sensor for this observation (abstract).
        aggregated_time(batch, use_custom_batching): Aggregate a batch of TensorObservations along the time axis.

    Raises:
        ValueError: If attempting to batch an empty list of observations.
        NotImplementedError: For methods that must be implemented by subclasses.
        AssertionError: If attempting to load or save data for batched observations (batch_size != 1).
    """

    container_name: str  # The name of this tensor observation container. Examples: 'rgb', 'depth', 'lidar', 'fisheye'
    _data: Optional[
        np.ndarray | torch.Tensor
    ]  # The data tensor of this sensor observation. Shape depends on the sensor type: (B>=1, T>=1, N>=1, ...)
    # None if loaded without payload. Access via property to automatically post-load it
    timestamps: np.ndarray | torch.Tensor  # The timestamps of each sensor observation. Shape: (B>=1, T>=1, N>=1)
    frame_ids: np.ndarray | torch.Tensor  # The integer frame ids of each sensor observation. Shape (B>=1, T>=1, N>=1)
    sensor_names: list[str]  # List of names for each sensor along the entity dimension (N). Shape: (N,)
    fileformat: str  # The default fileformat for serializing a single observation in data, e.g., jpg or bin.
    base_data_folder: str  # The base data folder for lazy loading of data

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

    @property
    def data(self) -> np.ndarray | torch.Tensor:
        """
        Returns the data, possibly lazy-loads data.

        DISCOURAGED to use this property directly, use specific properties from sub-classes instead.
        """
        if not self.in_memory():
            self.load_data()
        assert self._data is not None
        return self._data

    def _check_data_dimensions_impl(self, ignore_list: List[str] | None = None):
        self._check_array_dim("_data", 3, None, ignore_list)
        self._check_array_dim("timestamps", 3, None, ignore_list)
        self._check_array_dim("frame_ids", 3, None, ignore_list)

    @classmethod
    def build_sensor_batch(cls, observations: list):
        """Constructs a batched TensorObservation from a list of individual TensorObservation instances.
        This method combines multiple TensorObservation objects by stacking them along the sensor axis,
        resulting in a single batched observation suitable for model input or further processing.

        Args:
            observations (list): A list of TensorObservation instances to be batched.

        Returns:
            TensorObservation: A new TensorObservation instance representing the batched observations.

        Raises:
            ValueError: If the input list of observations is empty.
            NotImplementedError: This method is not yet implemented.
        """
        if len(observations) < 1:
            raise ValueError("Cannot batch an empty list of observations.")
        raise NotImplementedError()

    def get_idx_filename_pairs(self, base_path: str | None = None) -> Dict[Tuple[int, int], str]:
        """Return {(t_idx, s_idx): filename} dict entries for each time- and sensor-batched observation.

        The filename is constructed using the following pattern:
        {container_name}/{sensor_name}/{frame_id}.{fileformat}

        Returns:
            List[Tuple[Tuple[int, int], str]]: A list of tuples, each containing a (t_idx, s_idx) index pair and the corresponding filename.

        Raises:
            AssertionError: If the batch size of the observation is not 1.
        """

        if base_path is not None:
            warnings.warn(
                "The argument base_path is deprecated and will be removed in future versions.", DeprecationWarning
            )

        batch_size, t_dim, s_dim = self.timestamps.shape
        assert batch_size == 1, f"Cannot compute filenames for batched observation."

        res = {}
        for t_idx, s_idx in product(range(t_dim), range(s_dim)):
            sensor_name = self.sensor_names[s_idx]
            frame_id = self.frame_ids[0, t_idx, s_idx]
            filename = os.path.join(
                self.container_name,
                sensor_name,
                f"{frame_id}.{self.fileformat}",
            )
            res[(t_idx, s_idx)] = filename
        return res

    def in_memory(self) -> bool:
        """
        Checks if the data associated with this observation is stored in memory.

        Returns:
            bool: True if the data is present in memory (CPU or GPU), False otherwise.
        """
        return self._data is not None

    def set_base_data_folder(self, base_path: str) -> None:
        self.base_data_folder = base_path

    def load_data(self, base_path: str | None = None) -> None:
        """
        Loads observation data from disk into CPU memory if not already loaded.
        This method checks if the data is already in memory. If not, it initializes
        the data structures and loads data fragments from files corresponding to each
        timestamp and sensor index. Only supports batch size of 1.

        Args:
            base_path: The base directory path where data files are stored.

        Raises:
            AssertionError: If the batch size is not equal to 1.
        """
        if base_path is not None:
            self.base_data_folder = base_path

        if self.in_memory():
            return

        batch_size, _, _ = self.timestamps.shape
        assert batch_size == 1, f"Cannot load data for batch_size != 1"

        self._init_data()
        for (t_idx, s_idx), filename in self.get_idx_filename_pairs().items():
            self._load_fragment(t_idx, s_idx, os.path.join(self.base_data_folder, filename))

    def save_data(self, base_path: str):
        """
        Saves the data payload of this tensor observation to disk.
        This method writes the in-memory data fragments of the tensor observation to disk,
        using the provided base path. If the observation is not in memory, the method returns
        without performing any action. Only single (non-batched) observations are supported.

        Args:
            base_path: The base directory path where the data fragments will be saved.

        Raises:
            AssertionError: If the observation is batched (i.e., batch_size != 1).
        """

        batch_size, _, _ = self.timestamps.shape
        assert batch_size == 1, f"Cannot save batched observation."
        ### TODO: just return when not in memory? Assuming that it was already written?
        # assert self.in_memory()
        if not self.in_memory():
            return
        for (t_idx, s_idx), filename in self.get_idx_filename_pairs().items():
            self._save_fragment(t_idx, s_idx, os.path.join(base_path, filename))

    def serialize_data_class(self, object_serializer, base_path: str) -> dict:
        """
        Serializes the data class attributes (excluding '_data') into a dictionary.
        This method saves the current data to the specified base path, then serializes
        all attributes of the class except for the '_data' attribute using an
        ObjectSerializer. The result is a dictionary mapping attribute names to their
        serialized representations.

        Args:
            base_path (...): The base path where data should be saved and used for serialization.

        Returns:
            dict: A dictionary containing the serialized representations of the class attributes,
                excluding the '_data' attribute.
        """

        self.save_data(base_path)
        res = {}

        for key in self.__dict__.keys():
            if key == "_data":
                continue
            res[key] = object_serializer.serialize_obj_rec_(self.__dict__[key])
        return res

    def unload_data(self) -> None:
        """
        Removes the data associated with this observation from memory (CPU/GPU).
        This method deletes the in-memory data, freeing up resources. It does not affect any
        serialized data stored on disk.

        Returns:
            None
        """
        if self.in_memory():
            del self._data
            self._data = None

    @abstractmethod
    def _init_data(self):
        """
        Allocates blank tensors for loading the data payload into memory.
        This method initializes the necessary data structures to hold the data payload.
        It asserts that the data is not already loaded in memory before allocation.
        Raises:
            AssertionError: If the data is already loaded in memory.
        """
        assert not self.in_memory()

    @abstractmethod
    def _load_fragment(self, t_idx, s_idx, filename):
        raise NotImplementedError()

    @abstractmethod
    def _save_fragment(self, t_idx, s_idx, filename):
        raise NotImplementedError()

    @abstractmethod
    def get_sensor_type(self):
        raise NotImplementedError()

    @classmethod
    def aggregated_time(cls, batch: list[Self], use_custom_batching: list | None = None) -> Self:
        if use_custom_batching is None:
            use_custom_batching = []
        """
        Aggregates a batch of TensorObservation instances over time.
        This method calls the superclass's `aggregated_time` method, optionally using custom batching
        for the "sensor_names" attribute, and ensures that the resulting object's `sensor_names`
        attribute is set to that of the first batch element.
        
        Args:
            batch (list[Self]): A list of TensorObservation instances to aggregate.
            use_custom_batching (list, optional): List of attribute names to use custom batching for.
                Defaults to None.
                
        Returns:
            Self: An aggregated TensorObservation instance with updated `sensor_names`.
        """

        res = super().aggregated_time(batch, use_custom_batching=use_custom_batching + ["sensor_names"])

        res.sensor_names = batch[0].sensor_names

        return res


@dataclass
class CameraObservation(TensorObservation):
    """Stores non-privileged, tensor-shaped camera observations from the world.
    This class encapsulates camera image data and associated metadata, such as extrinsic and intrinsic
    calibration matrices, image dimensions, and sensor information. It supports batching and aggregation
    of observations along different axes (batch, time, sensor), and provides utilities for loading and
    saving image fragments.

    Attributes:
        extrinsics (np.ndarray | torch.Tensor): The homogeneous extrinsic transformation cam_T_vehicle
            for each observation. Shape: (B>=1, T>=1, N>=1, 4, 4).
        intrinsics (np.ndarray | torch.Tensor): The homogeneous intrinsic transformation img_T_cam
            for each observation. Shape: (B>=1, T>=1, N>=1, 3, 4).
        width (int): The image width in pixels. Shared among all observations in this container.
        height (int): The image height in pixels. Shared among all observations in this container.
        _dtype (type): The default dtype assumed for (de)serializing the data payload (default: np.uint8).

    Properties:
        rgb (np.ndarray | torch.Tensor): Returns the RGB tensor. Shape: (B>=1, T>=1, N>=1, 3, width, height).

    Methods:
        _init_data(): Allocates blank tensors for loading the images into memory.
        _save_fragment(t_idx, s_idx, filename): Saves a single image fragment to disk.
        _load_fragment(t_idx, s_idx, filename): Loads a single image fragment from disk.
        get_sensor_type(): Returns the sensor type (SensorTypes.CAMERA_RGB).
        create_empty(): Creates an empty CameraObservation instance (for compatibility).
        create_from_dict(obj_dict): Instantiates a CameraObservation from a dictionary.
        build_sensor_batch(observations): Batches a list of CameraObservation objects along the sensor axis.
        __getitem__(idx): Returns a sliced CameraObservation along the batch, time, or sensor axis.

    Raises:
        ValueError: If batching constraints are violated or unsupported configurations are encountered.
    """

    extrinsics: (
        np.ndarray | torch.Tensor
    )  # The homogeneous extrinsic transformation cam_T_vehicle for each observation. Shape: (B>=1, T>=1, N>=1, 4, 4)    ### TODO: Can this change over time?
    intrinsics: (
        np.ndarray | torch.Tensor
    )  # The homogeneous intrinsic transformation img_T_cam for each observation. Shape: (B>=1, T>=1, N>=1, 3, 4)     ### TODO: Can this change over time?
    width: int  # The image width in pixels. To enable batching, this value is shared among all observations in this container
    height: int  # The image height in pixels. To enable batching, this value is shared among all observations in this container
    _dtype = np.uint8  # The default dtype assumed for (de)serializing the data payload.

    @property
    def rgb(self) -> np.ndarray | torch.Tensor:
        """
        Returns the RGB tensor associated with this observation.
        This method is a simple wrapper around the `data` property and returns the RGB tensor.
        The expected shape of the tensor is (B>=1, T>=1, N>=1, 3, width, height).

        Returns:
            np.ndarray | torch.Tensor: The RGB tensor with shape (B, T, N, 3, width, height).

        Raises:
            AssertionError: If the data is not loaded in memory.
        """
        # assert self.in_memory()
        return self.data

    def _init_data(self):
        """
        Initializes and allocates blank tensors for image data storage.
        This method overrides the parent class's `_init_data` method. It creates a NumPy array
        to hold image data in memory, with dimensions based on the shape of `self.timestamps`
        and the specified image height and width. The resulting tensor has the shape
        (batch_size, t_dim, s_dim, 3, height, width), where 3 corresponds to the color channels.

        Raises:
            AttributeError: If `self.timestamps`, `self.height`, or `self.width` are not defined.
        """
        super()._init_data()
        batch_size, t_dim, s_dim = self.timestamps.shape
        self._data = np.zeros((batch_size, t_dim, s_dim, 3, self.height, self.width), dtype=self._dtype)

    def _save_fragment(self, t_idx, s_idx, filename):
        """
        Saves a fragment of the RGB data as an image file.
        This method extracts a specific fragment from the `self.rgb` array using the provided
        time index (`t_idx`) and spatial index (`s_idx`), checks its type and dtype, and saves
        it as an image file at the specified `filename` location. The parent directory is created
        if it does not exist.

        Args:
            t_idx (int): The time index to select the fragment from the RGB data.
            s_idx (int): The spatial index to select the fragment from the RGB data.
            filename (str or Path): The path where the image file will be saved.

        Raises:
            ValueError: If the fragment is a torch.Tensor (de-serialization not supported).
            ValueError: If the fragment's dtype is not np.uint8 (de-serialization not supported).
        """

        pathlib.Path(filename).parent.mkdir(exist_ok=True, parents=True)
        fragment = self.rgb[0, t_idx, s_idx]

        if isinstance(fragment, torch.Tensor):
            raise ValueError(f"De-serialization is currently not supported for data of type {fragment}")
        if fragment.dtype != np.uint8:
            raise ValueError(f"De-serialization is currently not supported for dtype={fragment.dtype}")

        rgb_img_arr = self.rgb[0, t_idx, s_idx].transpose(1, 2, 0)  # convert to H,W,3
        bgr_img_arr = cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_img_arr)

    def _load_fragment(self, t_idx, s_idx, filename):
        """
        Loads an image fragment from a file, transposes its axes, and stores it in the data array.

        Args:
            t_idx (int): Temporal index indicating the time step for storing the fragment.
            s_idx (int): Spatial index indicating the spatial position for storing the fragment.
            filename (str): Path to the image file to be loaded.

        Raises:
            FileNotFoundError: If the specified image file does not exist or cannot be read.
            AttributeError: If 'self._data' is not properly initialized or does not support assignment.
        """
        fragment = fragment_cache.query(filename)

        if fragment is None:
            img_arr = cv2.imread(filename)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            fragment = img_arr.transpose(2, 0, 1)  # convert to 3, H, W
            fragment_cache.set(filename, fragment)
        self._data[0, t_idx, s_idx] = fragment

    def get_sensor_type(self):
        """
        Returns the type of sensor used.

        Returns:
            SensorTypes: The type of sensor, specifically SensorTypes.CAMERA_RGB.
        """
        return SensorTypes.CAMERA_RGB

    @classmethod
    def create_empty(cls) -> Self:
        """
        Creates and returns an empty instance of the class for compatibility purposes.
        This method initializes an instance of the class with default or empty values for all attributes.
        It is intended to be used only for compatibility and may be removed in the future.

        Returns:
            CO: An instance of the class with all fields set to their default empty values.
        """

        # TODO: exists only for compatibility, remove in the future. As such suppress typing issues
        return cls(
            is_valid=None,  # type: ignore
            container_name="",
            _data=None,
            timestamps=None,  # type: ignore
            frame_ids=None,  # type: ignore
            sensor_names=[],
            fileformat="png",
            base_data_folder="",
            extrinsics=None,  # type: ignore
            intrinsics=None,  # type: ignore
            width=0,
            height=0,
        )

    @classmethod
    def create_from_dict(cls: Type[CO], obj_dict: dict) -> CO:
        """
        Creates an instance of the class from a dictionary by assigning its keys as attributes.

        Args:
            obj_dict (dict): A dictionary containing attribute names and their corresponding values.

        Returns:
            CO: An instance of the class with attributes set according to the provided dictionary.
        """

        res = cls.create_empty()
        for key in obj_dict.keys():
            res.__dict__[key] = obj_dict[key]
        return res

    @classmethod
    def build_sensor_batch(cls: Type[CO], observations: list[CO]) -> CO:
        """Constructs a batched sensor observation from a list of individual observations.
        This method combines multiple single-sensor observations into a single batched observation
        along the sensor axis. All input observations must have the same container name, width, and
        height, and must be loaded into memory. Each observation must represent a single batch, time,
        and sensor dimension, and all sensor names must be unique.

        Args:
            cls (Type[CO]): The class type to instantiate for the batched observation.
            observations (list[CO]): A list of individual sensor observations to batch.

        Returns:
            CO: A new batched observation containing data from all input observations.

        Raises:
            ValueError: If the input list is empty.
            ValueError: If observations belong to different containers.
            ValueError: If observations have different image widths or heights.
            ValueError: If any observation has batch_size > 1, t_dim > 1, or s_dim > 1.
            ValueError: If any observation is not loaded into memory.
            ValueError: If sensor names are not unique across observations.

        """
        if len(observations) < 1:
            raise ValueError("Cannot batch an empty list of observations.")
        ref_obs = observations[0]

        # check generic batching constraints
        if any(map(lambda o: o.container_name != ref_obs.container_name, observations)):
            raise ValueError("Cannot batch observations that belong to different containers.")
        if any(map(lambda o: o.width != ref_obs.width, observations)):
            raise ValueError("Cannot batch images with different width.")
        if any(map(lambda o: o.height != ref_obs.height, observations)):
            raise ValueError("Cannot batch images with different height.")

        # check for unsupported batching configurations
        if any(map(lambda o: o.timestamps.shape[0] != 1, observations)):
            raise ValueError("Cannot batch observations with batch_size > 1.")
        if any(map(lambda o: o.timestamps.shape[1] != 1, observations)):
            raise ValueError("Cannot batch observations with t_dim > 1.")
        if any(map(lambda o: o.timestamps.shape[2] != 1, observations)):
            raise ValueError("Cannot batch observations with s_dim > 1.")
        if any(map(lambda o: not o.in_memory(), observations)):
            raise ValueError("Cannot batch observations that have not been loaded into memory yet.")

        # check whether all sensor names are unique
        all_names = set(map(lambda o: o.sensor_names[0], observations))
        if len(all_names) < len(observations):
            raise ValueError("Cannot batch observations with identical sensor names.")

        # sort observations by sensor name
        observations = sorted(observations, key=lambda o: o.sensor_names[0])

        return cls(
            is_valid=np.concatenate(list(map(lambda o: o.is_valid, observations)), axis=2),
            container_name=ref_obs.container_name,
            _data=np.concatenate(list(map(lambda o: o.data, observations)), axis=2),
            timestamps=np.concatenate(list(map(lambda o: o.timestamps, observations)), axis=2),
            frame_ids=np.concatenate(list(map(lambda o: o.frame_ids, observations)), axis=2),
            sensor_names=[o.sensor_names[0] for o in observations],
            fileformat=ref_obs.fileformat,
            base_data_folder=ref_obs.base_data_folder,
            extrinsics=np.concatenate(list(map(lambda o: o.extrinsics, observations)), axis=2),
            intrinsics=np.concatenate(list(map(lambda o: o.intrinsics, observations)), axis=2),
            width=ref_obs.width,
            height=ref_obs.height,
        )

    def __getitem__(self, idx: int | Tuple, use_custom_slicing: List[str] | None = None) -> Self:
        """
        Retrieves a CameraObservation object for the specified index.
        This method normalizes the provided index and returns a new CameraObservation
        instance containing the corresponding data, timestamps, frame IDs, extrinsics,
        and intrinsics. Other attributes are passed through unchanged.

        Args:
            idx (int, or tuple): Index or indices specifying which observation(s) to retrieve.

        Returns:
            CameraObservation: An object containing the selected observation data.

        Raises:
            IndexError: If the index is out of bounds.
        """

        idx = self._normalize_idx(idx, 2)
        return type(self)(
            is_valid=self.is_valid[idx],
            container_name=self.container_name,
            _data=None if self._data is None else self._data[idx],
            timestamps=self.timestamps[idx],
            frame_ids=self.frame_ids[idx],
            sensor_names=self.sensor_names,
            fileformat=self.fileformat,
            base_data_folder=self.base_data_folder,
            extrinsics=self.extrinsics[idx],
            intrinsics=self.intrinsics[idx],
            width=self.width,
            height=self.height,
        )


@dataclass
class PointCloudObservation(TensorObservation):
    """Stores non-privileged, tensor-shaped pointclouds and associated metadata.
    This class manages batched point cloud data, including extrinsic calibration,
    point counts, and feature dimensions. It supports loading, saving, batching, and
    aggregation of point cloud observations, with support for both NumPy and PyTorch tensors.

    Attributes:
        extrinsics (np.ndarray | torch.Tensor): The homogeneous extrinsic transformation
            matrices (sensor_T_vehicle) for each observation. Shape: (B>=1, T>=1, N>=1, 4, 4).
        num_points (np.ndarray | torch.Tensor): The number of valid points in each
            observation. Shape: (B>=1, T>=1, N>=1).
        max_num_points (int): The maximum number of points per point cloud, used for batching.
            This number should be given based on the active sensor (radar/lidar) specifications.
            So it is a constant number across batch and time. If this number is set as negative,
            the maximum number of points across the batch and time will be calculated.
        num_features (int): The feature dimension of each point.
        _dtype (type): The default dtype assumed for (de)serializing the data payload.

    Properties:
        point_cloud (np.ndarray | torch.Tensor): Returns the point cloud tensor for the observation.

    Methods:
        _init_data():
            Allocates blank tensors for loading the point clouds into memory.
        get_sensor_type():
            Returns the sensor type (SensorTypes.LIDAR).
        _load_fragment(t_idx, s_idx, filename):
            Loads a fragment of point cloud data from disk.
        _save_fragment(t_idx, s_idx, filename):
            Saves a fragment of point cloud data to disk.
        create_empty():
            Creates an empty LidarObservation instance (for compatibility).
        create_from_dict(obj_dict):
            Instantiates a LidarObservation from a dictionary.
        build_sensor_batch(observations):
            Batches a list of LidarObservation instances along the sensor axis.
        _aggregated(agg_list, dim):
            Aggregates a list of LidarObservation instances along a specified dimension.
        __getitem__(idx):
            Returns a sliced LidarObservation instance.

    Raises:
        ValueError: If batching constraints are violated or unsupported configurations are encountered.
        NotImplementedError: If an unsupported file format is specified.
    """

    extrinsics: (
        np.ndarray | torch.Tensor
    )  # The homogeneous extrinsic transformation sensor_T_vehicle for each observation. Shape: (B>=1, T>=1, N>=1, 4, 4)
    num_points: (
        np.ndarray | torch.Tensor
    )  # The number of valid points in the observation. Required as all point_clouds are zero-padded to max_num_points. Shape: (B>=1, T>=1, N>=1)
    max_num_points: int  # The maximum number of points per point cloud. Required for batching. This number should be given based on the active sensor (radar/lidar) specifications or set to negative to auto-compute. So it is a constant number across batch and time.
    num_features: int  # The feature dimension of each point.
    _dtype = np.float32  # The default dtype assumed for (de)serializing the data payload.

    @property
    def point_cloud(self) -> np.ndarray | torch.Tensor:
        """
        Returns the point cloud tensor.
        This method provides access to the point cloud data stored in memory. It asserts that the data is loaded in memory before returning it.

        Returns:
            np.ndarray | torch.Tensor: The point cloud data as a NumPy array or PyTorch tensor.

        Raises:
            AssertionError: If the data is not loaded in memory.
        """
        return self.data

    def transform_to(self, m):
        ### TODO: Make it work with batch dim!
        assert self.shape[0] == 1, "For now transforming batched data is not supported! "

        new_data = self.data.copy()
        ts = list(self.data.shape)
        ts[-1] = 1
        hom_data = np.concatenate((self.data[..., :3], np.ones(ts)), axis=len(ts) - 1)
        ### TODO: Change to use "np.einsum("...ij,...nj -> ...ni", m, hom_data)" instead
        new_data[..., :3] = (hom_data @ np.moveaxis(m, -1, -2))[..., :3]

        res = LidarObservation(
            is_valid=self.is_valid,
            container_name=self.container_name,
            _data=new_data,
            timestamps=self.timestamps,
            frame_ids=self.frame_ids,
            sensor_names=self.sensor_names,
            fileformat=self.fileformat,
            base_data_folder=self.base_data_folder,
            extrinsics=self.extrinsics,
            num_points=self.num_points,
            max_num_points=self.max_num_points,
            num_features=self.num_features,
        )
        return res

    def transform_ego_centric(self):
        ### Extrinsics describe the transform from sensor to ego vehicle coordinates!
        return self.transform_to(self.extrinsics)

    def _init_data(self):
        """
        Initializes and allocates blank tensors for image data storage.
        This method creates a zero-initialized NumPy array to hold image data in memory,
        with dimensions based on the batch size, time steps, sensor count, maximum number
        of points, and number of features. It assumes that the shape of `self.timestamps`
        is (batch_size, t_dim, s_dim). The data is stored in `self.data` with the specified
        data type.

        Raises:
            AssertionError: If time-batching or sensor-batching is enabled (currently not supported).
        """
        super()._init_data()
        batch_size, t_dim, s_dim = self.timestamps.shape
        ### TODO: Why not?
        # assert t_dim == 1, "Time-batching is currently not supported"
        ### TODO: Why not?
        # assert s_dim == 1, "Sensor-batching is currently not supported"
        if self.max_num_points < 0:
            self.max_num_points = np.max(self.num_points)
        self._data = np.empty(
            (
                batch_size,
                t_dim,
                s_dim,
                self.max_num_points,
                self.num_features,
            ),
            dtype=self._dtype,
        )
        self._data.fill(np.nan)

    def _load_fragment(self, t_idx, s_idx, filename):
        """
        Loads a data fragment from a file and stores it in the internal data structure.
        Depending on the file format specified by `self.fileformat`, this method loads
        a fragment from either a `.npy` or `.npz` file and assigns it to the appropriate
        location in `self.data`.

        Args:
            t_idx (int): The time index at which to store the fragment.
            s_idx (int): The sample index at which to store the fragment.
            filename (str): The path to the file containing the fragment.

        Raises:
            NotImplementedError: If the file format specified is not supported.
        """
        fragment = fragment_cache.query(filename)

        if fragment is None:
            if self.fileformat == "npy":
                fragment = np.load(filename)
            elif self.fileformat == "npz":
                archiv = np.load(filename)
                fragment = archiv["pc"]
            else:
                raise NotImplementedError()
            fragment_cache.set(filename, fragment)

        if fragment.shape[0] > self._data[0, t_idx, s_idx].shape[0]:
            if self._data.shape[1] == 1 and self._data.shape[2] == 1:
                self._data = np.expand_dims(fragment, (0, 1, 2))
                return
            else:
                logger.warning(
                    "PointCloud data dimension do not match fragment: "
                    + str(fragment.shape[0])
                    + " > "
                    + str(self._data[0, t_idx, s_idx].shape[0])
                    + " Will re-allocate!!"
                )
                new_data = np.empty((1, self._data.shape[1], self._data.shape[2], fragment.shape[0], fragment.shape[1]))
                new_data.fill(np.nan)
                new_data[:, :, :, : self._data.shape[3], : self._data.shape[4]] = self._data
                self._data = new_data

        self._data[0, t_idx, s_idx] = fragment

    def _save_fragment(self, t_idx, s_idx, filename):
        """
        Saves a fragment of the point cloud to a file in the specified format.

        Args:
            t_idx (int): The time index of the fragment to save.
            s_idx (int): The sample index of the fragment to save.
            filename (str or Path): The file path where the fragment will be saved.

        Raises:
            ValueError: If the fragment is a torch.Tensor, as de-serialization is not supported.
            NotImplementedError: If the specified file format is not supported.

        Notes:
            Supported file formats are "npy" and "npz". The parent directory of the filename
            will be created if it does not exist.
        """

        pathlib.Path(filename).parent.mkdir(exist_ok=True, parents=True)
        fragment = self.point_cloud[0, t_idx, s_idx]

        if isinstance(fragment, torch.Tensor):
            raise ValueError(f"De-serialization is currently not supported for data of type {fragment}")
        # if fragment.dtype != self._dtype:
        #    raise ValueError(f"De-serialization is currently not supported for dtype={fragment.dtype}")

        if self.fileformat == "npy":
            np.save(filename, fragment)
        elif self.fileformat == "npz":
            np.savez_compressed(filename, pc=fragment)
        else:
            raise NotImplementedError()

    @classmethod
    def create_empty(cls) -> Self:
        """
        Creates and returns an empty instance of the class with default values.
        This method exists only for compatibility purposes and may be removed in the future.

        Returns:
            LO: An instance of the class with all fields set to their default empty values.
        """

        # TODO: exists only for compatibility, remove in the future. As such suppress typing issues
        return cls(
            is_valid=None,  # type: ignore
            container_name="",
            _data=None,
            timestamps=None,  # type: ignore
            frame_ids=None,  # type: ignore
            sensor_names=[],
            fileformat="png",
            base_data_folder="",
            extrinsics=None,  # type: ignore
            max_num_points=0,
            num_points=None,  # type: ignore
            num_features=0,
        )

    @classmethod
    def create_from_dict(cls, obj_dict: dict) -> Self:
        """
        Creates an instance of the class from a dictionary by assigning its keys as attributes.

        Args:
            obj_dict (dict): A dictionary containing attribute names and their corresponding values.

        Returns:
            An instance of the class with attributes set according to the provided dictionary.
        """

        res = cls.create_empty()
        for key in obj_dict.keys():
            res.__dict__[key] = obj_dict[key]
        return res

    @classmethod
    def build_sensor_batch(cls, observations: list[Self]) -> Self:
        """Constructs a batched PoPointCloudObservationint from a list of individual PointCloudObservation along the sensor axis.
        This method validates that all input observations are compatible for batching, including checks for
        container name, data shape, batch size, time dimension, sensor dimension, memory status, and unique
        sensor names. Observations are sorted by sensor name before batching.

        Args:
            cls (Self): The class type of the PointCloudObservation.
            observations (list[Self]): A list of PointCloudObservation instances to batch.

        Returns:
            Self: A new PointCloudObservation instance representing the batched observations.

        Raises:
            ValueError: If observations belong to different containers.
            ValueError: If observations have different data shapes.
            ValueError: If any observation has batch_size > 1.
            ValueError: If any observation has t_dim > 1.
            ValueError: If any observation has s_dim > 1.
            ValueError: If any observation is not loaded into memory.
            ValueError: If sensor names are not unique among observations.

        """
        ref_obs = observations[0]
        # if max_num_points is negavtive, set it to the max of the num points across all observations
        if ref_obs.max_num_points < 0:
            max_num_points = np.max([np.max(x.num_points) for x in observations])
        else:
            # else, set it to the max of the max_num_points across all observations
            max_num_points = np.max([x.max_num_points for x in observations])

        # check generic batching constraints
        if any(map(lambda o: o.container_name != ref_obs.container_name, observations)):
            raise ValueError("Cannot batch observations that belong to different containers.")

        # check for unsupported batching configurations
        if any(map(lambda o: o.timestamps.shape[0] != 1, observations)):
            raise ValueError("Cannot batch observations with batch_size > 1.")
        if any(map(lambda o: o.timestamps.shape[1] != 1, observations)):
            raise ValueError("Cannot batch observations with t_dim > 1.")
        if any(map(lambda o: o.timestamps.shape[2] != 1, observations)):
            raise ValueError("Cannot batch observations with s_dim > 1.")

        # check whether all sensor names are unique
        all_names = set(map(lambda o: o.sensor_names[0], observations))
        if len(all_names) < len(observations):
            raise ValueError("Cannot batch observations with identical sensor names.")

        # sort observations by sensor name
        observations = sorted(observations, key=lambda o: o.sensor_names[0])
        batched_data = cls._concat_tensors(
            [o.data for o in observations], 2, max_dim=[0, 0, 0, max_num_points, 0], fill_up_to_max=True
        )
        num_points = np.concatenate(list(map(lambda o: o.num_points, observations)), axis=2)

        return cls(
            is_valid=np.concatenate(list(map(lambda o: o.is_valid, observations)), axis=2),
            container_name=ref_obs.container_name,
            _data=batched_data,
            timestamps=np.concatenate(list(map(lambda o: o.timestamps, observations)), axis=2),
            frame_ids=np.concatenate(list(map(lambda o: o.frame_ids, observations)), axis=2),
            sensor_names=[o.sensor_names[0] for o in observations],
            fileformat=ref_obs.fileformat,
            base_data_folder=ref_obs.base_data_folder,
            extrinsics=np.concatenate(list(map(lambda o: o.extrinsics, observations)), axis=2),
            max_num_points=max_num_points,
            num_features=ref_obs.num_features,
            num_points=num_points,
        )

    @classmethod
    def aggregated_time(cls, batch: list[Self], use_custom_batching: List[str] | None = None) -> Self:
        """
        Aggregate a batch of PointCloudObservation instances along the time axis.

        This method combines multiple observations by concatenating their tensors along the time
        dimension, with special handling for point cloud-specific attributes.

        Args:
            batch (list[Self]): A list of PointCloudObservation instances to aggregate.
            use_custom_batching (list, optional): List of attribute names to use custom batching for.
                Defaults to None.

        Returns:
            An aggregated PointCloudObservation instance.
        """
        # Let parent class handle common aggregation, excluding _data and max_num_points for custom handling
        res = super().aggregated_time(batch, ["_data", "max_num_points"])

        # Custom batching for `max_num_points`: take the maximum over the batch.
        # This ensures the resulting observation can accommodate all points from any timestep.
        # The parent's aggregation would simply copy the first batch entry's value, which
        # could be insufficient for concatenated point clouds with varying point counts.
        if batch[0].max_num_points < 0:
            res.max_num_points = np.max([np.max(x.num_points) for x in batch])
        else:
            res.max_num_points = np.max([x.max_num_points for x in batch])

        # Handle concatenation of point cloud data tensors along time axis
        if np.all([x._data is not None for x in batch]):
            res._data = super()._concat_tensors(
                [x._data for x in batch if x._data is not None],
                1,
                max_dim=[0, 0, 0, res.max_num_points, 0],
                fill_up_to_max=True,
            )
        else:
            # Ensure consistent handling: all observations must be either loaded or unloaded
            if np.any([x._data is not None for x in batch]):
                raise ValueError(
                    "Mixing empty Tensor Observations with data==None with ones with data is not supported."
                )

        return res

    def __getitem__(self, idx: int | Tuple, use_custom_slicing: List[str] | None = None) -> Self:
        """
        Retrieves a LidarObservation instance for the given index.

        Args:
            idx (int or tuple): Index or slice specifying which observation(s) to retrieve.

        Returns:
            LidarObservation: An instance containing the data and metadata for the specified index.

        Raises:
            IndexError: If the index is out of range.
        """

        idx = self._normalize_idx(idx, 2)
        return self.__class__(
            is_valid=self.is_valid[idx],
            container_name=self.container_name,
            _data=None if self._data is None else self._data[idx],
            timestamps=self.timestamps[idx],
            frame_ids=self.frame_ids[idx],
            sensor_names=self.sensor_names,
            fileformat=self.fileformat,
            base_data_folder=self.base_data_folder,
            extrinsics=self.extrinsics[idx],
            num_points=self.num_points[idx],
            max_num_points=self.max_num_points,
            num_features=self.num_features,
        )


@dataclass
class LidarObservation(PointCloudObservation):
    """Stores non-privileged, tensor-shaped lidar pointclouds and associated metadata.
    For more details see PointCloudObservation
    """

    def get_sensor_type(self):
        """
        Returns the type of sensor associated with this observation.

        Returns:
            SensorTypes: The sensor type, specifically SensorTypes.LIDAR.
        """

        return SensorTypes.LIDAR


@dataclass
class RadarObservation(PointCloudObservation):
    """Stores non-privileged, tensor-shaped radar pointclouds.
    For more details see PointCloudObservation
    """

    def get_sensor_type(self):
        """
        Returns the type of sensor.

        Returns:
            SensorTypes: The type of sensor, specifically SensorTypes.RADAR.
        """

        return SensorTypes.RADAR
