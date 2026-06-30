import numpy as np

from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.data_interface.episode_map import EpisodeMapUtils, MapContainer
from bevad_sim.data_interface.odometry import Odometry
from bevad_sim.data_interface.tce import TrafficControlElementsUtils
from bevad_sim.data_interface.tensor_observation import CameraObservation, LidarObservation, RadarObservation
from bevad_sim.data_interface.transforms.base_transform import BaseTransform
from bevad_sim.data_interface.transforms.util_transform import invert_transform
from bevad_sim.data_interface.world_state import WorldStateUtils


class Make2D(BaseTransform):
    """Transform that projects world state and maps to a 2D representation.

    Sets all z-coordinates in the world state and episode maps to zero, producing
    a 2D view while retaining the CoreContainer structure.
    """

    def __call__(self, container: CoreContainer) -> CoreContainer:
        """Apply 2D projection to the world state and map data in the container.

        Mutates the input container by zeroing out the z-axis of the world state
        and each EpisodeMap in the map container.

        Args:
            container: A CoreContainer instance containing world_state and map_container.

        Returns:
            The same CoreContainer instance with 2D-projected world_state and maps.
        """
        assert container.world_state is not None
        WorldStateUtils.make_2d(container.world_state)
        # mypy: container.map_container could be None, only MapContainer has maps
        if container.map_container is not None:
            for m in container.map_container.maps:
                EpisodeMapUtils.make_2d(m)
        return container


class MakeEgoCentricOld(BaseTransform):
    """Transform that converts world state and maps to an ego-centric coordinate frame.

    Uses the ego vehicle's pose at a reference time t0 to compute an inverse transform,
    then applies it to world_state, episode maps, and traffic control elements so that
    the ego vehicle is at the origin.

    Attributes:
        t0: Reference time index at which to compute the ego pose (default 0).
        only_position: If True, uses only the ego position (ignores orientation) when
                       building the inverse transform; otherwise uses full pose transform.
    """

    def __init__(self, t0: int = 0, only_position: bool = False):
        """Initialize the ego-centric transform.

        Args:
            t0: Time step to use as reference for ego pose. Defaults to 0.
            only_position: If True, only translate position without applying rotation
                transformation. Useful when you want to preserve global orientation
                but center coordinates on ego vehicle. Defaults to False.
        TODO:
            Change the name of the argument 'only_position' to 'only_translation'
        """

        self.t0 = t0
        self.only_position = only_position

    def __call__(self, container: CoreContainer) -> CoreContainer:
        """Transform container to ego-centric coordinates.

        This method extracts the ego vehicle's pose at the specified time step
        and transforms all coordinate-dependent data in the container to be
        relative to the ego vehicle's position and orientation.

        Args:
            container: Input container with world state data including ego vehicle
                information, maps, traffic control elements, and other scene data.

        Returns:
            A new CoreContainer with all coordinate data transformed to ego-centric
            reference frame. The ego vehicle will be at the origin with identity
            orientation (if only_position=False) or at origin with preserved
            global orientation (if only_position=True).

        """

        ## TODO: should it work on batched data? Yes, it should

        # Create a new, empty CoreContainer to accumulate transformed data
        res = CoreContainer.create_empty()
        res.episode_meta = container.episode_meta
        ## TODO: do we need to transform also ego poses for the measurements?

        # Retrieve ego index and state at reference time t0
        # mypy: container.world_state could be None, only WorldState has ego_id
        if container.world_state is None or container.world_state.ego_id is None:
            raise AttributeError("container.world_state or ego_id is None")
        ego_id = container.world_state.ego_id.item()
        ego_idx = WorldStateUtils.get_idx_by_id(container.world_state, ego_id)
        ego_state = WorldStateUtils.get_state_by_idx(container.world_state, b=0, t=self.t0, n=ego_idx)

        # Determine inverse transform: full pose or position-only
        if not self.only_position:
            assert isinstance(ego_state, np.ndarray), "currently only support numpy"
            inv_state = invert_transform(ego_state)
        else:
            inv_state = np.eye(4)
            inv_state[0, 3] = -ego_state[0, 3]
            inv_state[1, 3] = -ego_state[1, 3]

        # Transform map container if present
        if container.map_container is not None:
            maps = [EpisodeMapUtils.transform(map, inv_state) for map in container.map_container.maps]
            res.map_container = MapContainer(maps=maps, is_valid=np.ones((len(maps))))

        # Transform world state to ego-centric frame
        res.world_state = container.world_state
        res.world_state.transform = WorldStateUtils.transform(container.world_state, inv_state)

        if container.tce is not None and isinstance(container.tce.transform, np.ndarray):
            res.tce = TrafficControlElementsUtils.transform_traffic_control_elements(container.tce, inv_state)

        res.routing_information = container.routing_information
        res.step_meta = container.step_meta
        res.odometry = container.odometry
        res.action = container.action
        res.metric = container.metric

        return res


class MakeEgoCentric(BaseTransform):
    def __init__(self, t0: int = -1, only_translation: bool = False):
        """Initialize the ego-centric transform.

        Args:
            t0: Time step to use as reference for ego pose. Defaults to 0.
            only_translation: If True, only translate position without applying rotation
                transformation. Useful when you want to preserve global orientation
                but center coordinates on ego vehicle. Defaults to False.
        """

        super().__init__()
        self.t0 = t0
        self.only_translation = only_translation

    def __call__(self, container: CoreContainer) -> CoreContainer:
        ## TODO: should it work on batched data? Yes, it should

        # Create a new, empty CoreContainer to accumulate transformed data
        ## TODO: do we need to transform also ego poses for the measurements?

        # Retrieve ego index and state at reference time t0
        if container.odometry is not None and container.odometry.transform is not None:
            ego_state = container.odometry.transform[:, self.t0]
        elif (
            container.world_state is not None
            and container.world_state.transform is not None
            and container.world_state.ego_id is not None
        ):
            ego_state = container.world_state.transform[
                :, self.t0, WorldStateUtils.get_idx_by_id(container.world_state, container.world_state.ego_id.item())
            ]
        else:
            raise ValueError(
                "Container must contain either odometry or world state to perform ego-centric transformation"
            )

        assert isinstance(ego_state, np.ndarray), "currently only support numpy"
        inv_state = invert_transform(ego_state)

        res = container.shallow_copy()

        # Transform map container if present
        if container.map_container is not None:
            emaps = []
            for i, emap in enumerate(container.map_container.maps):
                emaps.append(emap.transform_to(inv_state[i]))
            res.map_container = MapContainer(container.map_container.is_valid, emaps)

        if container.world_state is not None:
            res.world_state = container.world_state.transform_to(inv_state)

        if container.tce is not None:
            res.tce = container.tce.transform_to(inv_state)

        if container.odometry is not None and container.odometry.transform is not None:
            res.odometry = container.odometry.transform_to(inv_state)

        if container.routing_information is not None:
            res.routing_information = container.routing_information.transform_to(inv_state)

        for key, data in container._filter_base_entity().items():
            if isinstance(data, CameraObservation):
                res.__dict__[key] = data
            if isinstance(data, LidarObservation):
                res.__dict__[key] = data.transform_ego_centric()
            if isinstance(data, RadarObservation):
                res.__dict__[key] = data.transform_ego_centric()

        return res


class WorldStateToOdometry(BaseTransform):
    """Transform that extracts odometry information from world state data.

    This transform creates odometry data from the ego vehicle's world state information
    when odometry is not directly available in the dataset.
    """

    def __init__(self):
        """Initialize the WorldState to Odometry transform."""
        pass

    def __call__(self, container: CoreContainer) -> CoreContainer:
        """Extract odometry from world state and add it to the container.

        This method extracts the ego vehicle's pose and dynamics from world state
        and creates corresponding odometry data structure.

        Args:
            container: Input container with world state data but potentially missing odometry.

        Returns:
            The same CoreContainer instance with odometry data populated from world state.
        """
        assert container.world_state is not None
        assert container.world_state.ego_id is not None
        assert container.world_state.transform is not None

        # Get ego vehicle information
        ego_id = container.world_state.ego_id.item()
        ego_idx = WorldStateUtils.get_idx_by_id(container.world_state, ego_id)

        # Create odometry structure if it doesn't exist
        if container.odometry is None:
            # Initialize odometry with the same structure as world state

            batch_size = container.world_state.transform.shape[0]
            time_steps = container.world_state.transform.shape[1]

            container.odometry = Odometry.create_zeros(b=batch_size, t=time_steps)

        # Extract ego transform and dynamics for all time steps
        # Shape: [batch_size, time_steps, 4, 4] for transform
        assert isinstance(container.world_state.transform, np.ndarray), "currently only support numpy"
        container.odometry.transform = container.world_state.transform[:, :, ego_idx, :, :].copy()

        # Extract speed from dynamics (assuming first component is longitudinal velocity)
        # Shape: [batch_size, time_steps]
        assert isinstance(container.world_state.dynamics, np.ndarray), "currently only support numpy"
        container.odometry.speed = container.world_state.dynamics[:, :, ego_idx, 0].copy()

        if hasattr(container.odometry, "acceleration"):
            assert isinstance(container.world_state.dynamics, np.ndarray), "currently only support numpy"
            container.odometry.acceleration = container.world_state.dynamics[:, :, ego_idx, 1].copy()

        return container


class FixEpisodeMetaFormat(BaseTransform):
    """
    Transform that ensures EpisodeMeta fields have correct types.

    This addresses legacy type issues from dataset conversions (e.g., nuplan)
    where scalar values need to be converted to arrays/lists for consistency
    with the expected data format in the replay simulator.
    """

    def __call__(self, container: CoreContainer) -> CoreContainer:
        if container.episode_meta is None:
            return container

        meta = container.episode_meta

        if meta.episode_id is not None and isinstance(meta.episode_id, str):
            meta.episode_id = [meta.episode_id]

        if meta.scenario_type is not None and isinstance(meta.scenario_type, str):
            meta.scenario_type = [meta.scenario_type]

        if meta.region is not None and isinstance(meta.region, str):
            meta.region = [meta.region]

        if meta.frame_rate is not None and isinstance(meta.frame_rate, (int, float)):
            meta.frame_rate = np.array([meta.frame_rate], dtype=np.float32)

        if meta.weather is not None and isinstance(meta.weather, str):
            meta.weather = [meta.weather]

        if container.world_state is not None:
            container.world_state.ego_id = np.array([container.world_state.ego_id])

        if container.step_meta is not None:
            container.step_meta.timestamps = np.array([container.step_meta.timestamps], dtype=np.float64)
            container.step_meta.frame_ids = np.array([container.step_meta.frame_ids], dtype=np.int64)

        return container
