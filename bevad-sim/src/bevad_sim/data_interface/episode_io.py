from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Optional

import numpy as np

from bevad_sim.data_interface.base_entity import BaseEntity
from bevad_sim.data_interface.core_container import CoreContainer
from bevad_sim.data_interface.serialize_objects import (
    IOUtils,
    ObjectDeSerializer,
    ObjectSerializer,
)
from bevad_sim.version import get_vcs_version


class EpisodeIo:
    """Handles saving and loading CoreContainer episodes to/from disk."""

    @classmethod
    def add_file_metadata(cls, payload, version_str) -> dict:
        return {
            "file_info": {
                "data_version": _CURRENT_FILE_VERSION.value,
                "bevad_sim_version": version_str,
            },
            "payload": payload,
        }

    @classmethod
    def write_episode(cls, folder: str | None, data: CoreContainer, config=None) -> None:
        """Writes a CoreContainer episode to disk.

        Data can be split into "core fields" (default) --> written to the specified folder
        and "custom fields" --> specified together with their location in the config.
        See tests for working example configs.

        Args:
            folder: Path to the output directory. Set to `None` to only save specified custom fields.
            data: The episode data to write.
            config: Configuration used during serialization, containing specification of custom fields.
                See tests for working example configs.
        """
        if config is None:
            config = EpisodeIo.default_config()
        # TODO: if B>1 then iterate over batch dim and save data in different episodes

        version = get_vcs_version()
        if "-" in version:
            print(
                f"WARNING: You are using a development version for serialization ({version}). This is higly discouraged!"
            )

        # Serialize the core container into memory, then write to disk.
        # Hint: TensorObservations will be dumped directly during serialization
        if folder is not None:
            IOUtils.check_and_create(folder)
            obj_ser = ObjectSerializer(folder)
            ignore_fields = config["serialize_custom_fields"].keys() if "serialize_custom_fields" in config else set()
            payload = obj_ser.serialize_obj_rec_(data, ignore_root_fields=ignore_fields)
            res = cls.add_file_metadata(payload, version)
            IOUtils.write_struct(res, folder + "/ep_data." + IOUtils.get_file_extension())

        # Serialize and write custom fields to disk, grouped + packed by folder:
        if "serialize_custom_fields" in config:
            data_fields = set(
                attr for attr in dir(data) if not callable(getattr(data, attr)) and not attr.startswith("__")
            )
            for custom_folder in set(filter(lambda x: x is not None, config["serialize_custom_fields"].values())):
                selected_fields = set(
                    field for field, folder in config["serialize_custom_fields"].items() if folder == custom_folder
                )
                IOUtils.check_and_create(custom_folder)
                obj_ser = ObjectSerializer(custom_folder)
                ignore_fields = data_fields - selected_fields - set(["episode_meta"])
                payload = obj_ser.serialize_obj_rec_(data, ignore_root_fields=ignore_fields)
                res = cls.add_file_metadata(payload, version)
                IOUtils.write_struct(res, custom_folder + f"/custom_fields.{IOUtils.get_file_extension()}")

    @classmethod
    def read_episode(
        cls, folder: str, load_payload: bool = True, config=None, class_path_hints: Optional[Dict] = None
    ) -> CoreContainer:
        """Read and deserialize episode data from the specified folder.

        The method searches for a file named 'ep_data.<extension>' in the given
        folder, reads the serialized structure using IOUtils, and reconstructs
        the CoreContainer instance with ObjectDeSerializer. Optionally, tensor
        observation payloads (e.g., images or point clouds) can be loaded.

        Args:
            folder: The directory path containing the serialized episode data.
            load_payload: If True, loads the payload of tensor observations;
                otherwise, payload data is skipped.
            config: Configuration used during deserialization. This may include
                a specification of custom fields to load from other locations.
                See tests for working example configs.
            class_path_hints: If not None, contains paths to classes for specified objects. Can either be a class directly or a path as str.

        Returns:
            A CoreContainer instance representing the deserialized episode data.

        Raises:
            FileNotFoundError: If no file matching 'ep_data.<extension>' is found.
            IOError: If there is an error reading or deserializing the file.
        """
        if config is None:
            config = EpisodeIo.default_config()

        # right now zstd is used as compression method.
        extens = "zstd"
        struct = IOUtils.read_struct(folder + "/ep_data." + extens)

        migrator = _VersionMigrator(struct)
        struct = migrator.migrate_raw_struct(struct)

        obj_ser = ObjectDeSerializer(folder, load_payload=load_payload, class_path_hints=class_path_hints)
        data = obj_ser.de_serialize_obj_rec_(struct["payload"])
        assert type(data) == CoreContainer

        data = migrator.migrate_core_container(data)

        # Read custom fields:
        if "serialize_custom_fields" in config:
            for custom_folder in set(filter(lambda x: x is not None, config["serialize_custom_fields"].values())):
                custom_struct = IOUtils.read_struct(custom_folder + "/custom_fields." + extens)
                custom_migrator = _VersionMigrator(custom_struct)
                custom_struct = custom_migrator.migrate_raw_struct(custom_struct)
                custom_obj_ser = ObjectDeSerializer(custom_folder, load_payload=load_payload)
                custom_data = custom_obj_ser.de_serialize_obj_rec_(custom_struct["payload"])
                assert data.episode_meta.episode_id == custom_data.episode_meta.episode_id
                for custom_field in (
                    field for field, folder in config["serialize_custom_fields"].items() if folder == custom_folder
                ):
                    setattr(data, custom_field, getattr(custom_data, custom_field))

        return data

    @classmethod
    def default_config(cls):
        """Returns default config for episode I/O.

        Returns:
            dict: A dictionary with default configuration values.
        """
        config = {}  # obsio.default_config()

        return config


class _FileVersion(Enum):
    """Distinct data version (int values) named after the introducing e2e-core version"""

    V010 = 1
    V020 = 2
    V032 = 3
    V050 = 4


_CURRENT_FILE_VERSION = _FileVersion.V050


class _VersionDetector:
    def detect(self, raw_struct) -> _FileVersion:
        if type(raw_struct) == dict and "file_info" in raw_struct:
            return _FileVersion(raw_struct["file_info"]["data_version"])
        if raw_struct[0] == "bevad_sim.common.data_structures.episode.Episode":
            return _FileVersion.V010
        elif raw_struct[0] == "bevad_sim.data_interface.core_container.CoreContainer":
            if "is_valid" not in raw_struct[1]:
                return _FileVersion.V020
            else:
                return _FileVersion.V032
        raise ValueError("File version cannot be detected, no supported base class found.")


class _VersionMigrator:
    def __init__(self, struct):
        self._detected_version = _VersionDetector().detect(struct)

    def migrate_raw_struct(self, struct):
        if self._detected_version == _FileVersion.V010:
            struct = EpisodeIo.add_file_metadata(self._migrate_raw_struct_010(struct), "dummy_version_by_migrator")
        if self._detected_version == _FileVersion.V020:
            struct = EpisodeIo.add_file_metadata(self._migrate_raw_struct_020(struct), "dummy_version_by_migrator")
        if self._detected_version == _FileVersion.V032:
            struct = EpisodeIo.add_file_metadata(struct, "dummy_version_by_migrator")
        return struct

    def migrate_core_container(self, cc: CoreContainer) -> CoreContainer:
        if self._detected_version == _FileVersion.V010:
            cc = self._migrate_core_container_010(cc)
        if self._detected_version == _FileVersion.V020:
            cc = self._migrate_core_container_020(cc)
        if self._detected_version == _FileVersion.V032:
            # shouldnt be needed, but carla observer missed to fill some is_valid flags
            # support already generated data for this version
            cc = self._migrate_core_container_032(cc)
        return cc

    @classmethod
    def _create_is_valid_for_base_entity(cls, be: BaseEntity):
        if be.is_valid is not None:
            be.is_valid = np.array(be.is_valid, dtype=bool)
            return
        for v in be.__dict__.values():
            if v is not None and isinstance(v, np.ndarray):
                if len(v.shape) >= be.dimensionality:
                    be.is_valid = np.ones(v.shape[: be.dimensionality], dtype=bool)
                    return

    @classmethod
    def _populate_is_valid_field(cls, cc: CoreContainer):
        for be in cc._filter_base_entity().values():
            cls._create_is_valid_for_base_entity(be)

        if cc.is_valid is not None:
            return
        if cc.world_state is not None:
            cc.is_valid = np.ones(cc.world_state.shape[:2], dtype=bool)
            return
        if cc.tce is not None:
            cc.is_valid = np.ones(cc.tce.shape[:2], dtype=bool)
            return
        if cc.odometry is not None:
            cc.is_valid = np.ones(cc.odometry.shape[:2], dtype=bool)
            return
        if cc.step_meta is not None:
            cc.is_valid = np.ones(cc.step_meta.shape[:2], dtype=bool)
            return

        for obs in cc.camera_observations.values():
            if obs is not None and obs.is_valid is not None:
                cc.is_valid = np.ones(obs.shape[:2], dtype=bool)
                return

        for obs in cc.lidar_observations.values():
            if obs is not None and obs.is_valid is not None:
                cc.is_valid = np.ones(obs.shape[:2], dtype=bool)
                return

    @classmethod
    def _create_missing_is_valid_dim(cls, cc: CoreContainer):
        for be in cc._filter_base_entity().values():
            # none check because migration logic undefined for MapContainer
            if be.is_valid is not None and len(be.is_valid.shape) < be.dimensionality:
                common_dims = max([v.shape for v in be.__dict__.values() if isinstance(v, np.ndarray)], key=len)[
                    : be.dimensionality
                ]
                for dim_idx in range(len(common_dims)):
                    if dim_idx < len(be.is_valid.shape):
                        continue
                    else:
                        be.is_valid = np.expand_dims(be.is_valid, -1)
                        be.is_valid = np.broadcast_to(be.is_valid, common_dims[: dim_idx + 1])
        return cc

    def _migrate_raw_struct_020(self, struct):
        return struct

    def _migrate_core_container_032(self, cc: CoreContainer):
        self._populate_is_valid_field(cc)
        return cc

    def _migrate_core_container_020(self, cc: CoreContainer):
        self._populate_is_valid_field(cc)
        self._create_missing_is_valid_dim(cc)
        return cc

    # it may be reasonable later to extract separate classes for each version one code complexity increases:
    def _migrate_raw_struct_010(self, struct):
        self._migrated_tensor_observations = []  # store for post-processing of CoreContainer
        assert set(struct[1].keys()) >= {
            "frame_ids",
            "timestamps",
            "tensor_observations",
            "worldstates",
            "odometry",
            "traffic_control_elements",
            "minimap",
            "episode_meta",
            "navigation_route",
        }
        # no 'rewards' but 'step_meta'
        migrated_struct = (
            "bevad_sim.data_interface.core_container.CoreContainer",
            {
                "episode_meta": (
                    "bevad_sim.common.data_structures.episode_meta.EpisodeMeta",
                    struct[1]["episode_meta"][1],
                ),
                "step_meta": (
                    "bevad_sim.common.data_structures.step_meta.StepMeta",
                    {
                        "frame_ids": struct[1]["frame_ids"],
                        "timestamps": struct[1]["timestamps"],
                    },
                ),
                "odometry": ("bevad_sim.data_interface.odometry.Odometry", struct[1]["odometry"][1]),
                "world_state": (
                    "bevad_sim.common.data_structures.world_state.WorldState",
                    struct[1]["worldstates"][1],
                ),
                # TODO: the following works but member .control_box is None in the example data, violating not-None assumptions
                #'tce': ('bevad_sim.common.data_structures.tce.TrafficControlElements', struct[1]['traffic_control_elements'][1]),
                # TODO: find old example with map and try to activate:
                #'map_container': ('bevad_sim.data_interface.episode_map.MapContainer', {'maps': struct[1]['minimap']}),
                "routing_information": (
                    "bevad_sim.common.data_structures.routing_information.RoutingInformation",
                    struct[1]["navigation_route"][1],
                ),
            },
        )
        migrated_struct[1]["world_state"][1]["ego_id"] = [
            migrated_struct[1]["episode_meta"][1].pop("ego_id")
        ]  # field was moved
        if "tce" in migrated_struct[1].keys():
            migrated_struct[1]["tce"][1]["tce_id"] = migrated_struct[1]["tce"][1].pop("id")  # field was renamed
        if "rewards" in struct[1].keys():
            migrated_struct[1]["step_meta"][1]["reward"] = struct[1]["rewards"]
        # move tensor observations one hierarchy level up, time-aggreate later after deserialization
        for k, v in struct[1]["tensor_observations"].items():
            # data field was renamed into a private field
            if type(v) == list:
                for to_tuple in v:
                    to_tuple[1]["_data"] = to_tuple[1].pop("data")
            migrated_struct[1][k] = v
            self._migrated_tensor_observations.append(k)
        return migrated_struct

    def _migrate_core_container_010(self, cc: CoreContainer) -> CoreContainer:
        def numpy_array(field):
            return np.array([field]) if isinstance(field, list) else field

        if cc.step_meta is not None:
            cc.step_meta.frame_ids = numpy_array(cc.step_meta.frame_ids)
            cc.step_meta.timestamps = numpy_array(cc.step_meta.timestamps)
            cc.step_meta.reward = numpy_array(cc.step_meta.reward)
        if cc.world_state is not None:
            cc.world_state.ego_id = numpy_array(cc.world_state.ego_id)
        for k in self._migrated_tensor_observations:
            if type(cc.__dict__[k]) == list:
                cc.__dict__[k] = cc.__dict__[k][0].aggregated_time(cc.__dict__[k])

        self._populate_is_valid_field(cc)

        return cc
