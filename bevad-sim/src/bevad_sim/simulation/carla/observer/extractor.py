from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import partial

import carla
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from bevad_sim.data_interface.tce import TrafficControlElements
from bevad_sim.data_interface.world_state import TransformsOperations, WorldState
from bevad_sim.simulation.carla.labels import (
    CARLA_ATTRIBUTE_MAP,
    CARLA_CATEGORY_MAP,
    get_attribute_from_blueprint,
    get_carla_traffic_light_state,
    get_category_from_blueprint,
    get_category_from_city_object_label,
    get_category_from_mesh,
    get_static_traffic_sign_type,
)
from bevad_sim.simulation.carla.observer.carla_map_converter import CarlaMapConverter
from bevad_sim.simulation.carla.utils import convert_carla_rotation, convert_carla_vector, convert_carla_vector_noflip

CALL_BUTTON_Z_THRESHOLD = 0.3
"""Heuristic for distinguishing call buttons from actual light boxes."""


def build_world_state_extractor(town: str, world):
    if town in ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"):
        return SmallMapWorldStateExtractor(world)
    elif town in ("Town15",):
        return Town15WorldStateExtractor(world)
    elif town in ("Town11", "Town12", "Town13"):
        return LargeMapWorldStateExtractor(world)
    else:
        raise ValueError(f"No world state extractor implemented for town '{town}'.")


class IdGenerator:
    """Generate unique IDs for CARLA actors and traffic control elements."""

    def __init__(self):
        self.id_map = {}

    def id_for_actor(self, actor):
        """Generate an ID for a generic carla.Actor instances (e.g., vehicles, pedestrians)."""
        key = (actor.id, -1)
        return self._id_for_key(key)

    def id_for_light_box(self, tl_actor, box_id: int):
        """Generate an ID for a traffic light box. Call this method with the carla.TrafficLight actor and the light box id."""
        key = (tl_actor.id, box_id)
        return self._id_for_key(key)

    def id_for_int_val(self, int_val: int):
        """Generate an ID for a generic integer value. Use this method for non-carla.Actor instances, such as environment objects."""
        key = (int_val, -2)
        return self._id_for_key(key)

    def _id_for_key(self, key: tuple[int, int]):
        """ "Returns the ID for a key. If no ID is registered for a key, a new ID is generated, registered and returned."""
        if not key in self.id_map:
            id = len(self.id_map)
            self.id_map[key] = id
        return self.id_map[key]


class CarlaWorldStateExtractor(ABC):
    """Extract privileged world state from CARLA."""

    def __init__(self, world):
        self.world = world
        self.idg = IdGenerator()

        # static signs that belong to the map
        self._cached_static_traffic_signs: list | None = None

    def extract_frame_snapshot(self):
        """Extract privileged world state for a snapshot. This method should be called once per tick."""
        world_snapshot = self.world.get_snapshot()
        timestamp = world_snapshot.timestamp.elapsed_seconds
        frame_id = world_snapshot.frame
        actors_dict = {a.id: a for a in self.world.get_actors()}

        # extract all object types
        dynamic_objects = self._extract_dynamic_objects(actors_dict)
        parked_vehicles = self._extract_parked_vehicles(actors_dict)
        traffic_lights = self._extract_traffic_lights(actors_dict)
        static_traffic_signs = self._extract_static_traffic_signs()

        # find the ego vehicle
        ego_vehicle = dynamic_objects[0]  # by design, ego is always present
        ego_id = ego_vehicle["track_id"]
        ego_pos = ego_vehicle["world_tf_box"][:3, 3]

        def distance_criterion(object, max_bev_distance: float = 100.0, max_z_distance: float = 10.0):
            pos = object["world_tf_box"][:3, 3]
            return (
                np.linalg.norm(ego_pos[:2] - pos[:2]) <= max_bev_distance
                and np.abs(ego_pos[2] - pos[2]) <= max_z_distance
            )

        # TODO: make thresholds configurable
        ob_dist_criterion = partial(distance_criterion, max_bev_distance=75, max_z_distance=10.0)
        tl_dist_criterion = partial(distance_criterion, max_bev_distance=75, max_z_distance=15.0)

        # apply a range filter on all extracted objects
        local_dynamic_objects = list(filter(ob_dist_criterion, dynamic_objects))
        local_parked_vehicles = list(filter(ob_dist_criterion, parked_vehicles))
        local_traffic_lights = list(filter(tl_dist_criterion, traffic_lights))
        local_static_signs = list(filter(tl_dist_criterion, static_traffic_signs))

        # determine the size of the world state
        all_objects = local_dynamic_objects + local_parked_vehicles

        # build world state
        ws = WorldState._create_zeros(t=1, n=len(all_objects))
        for i, o in enumerate(all_objects):
            ws.transform[0, 0, i, :, :] = o["world_tf_box"]
            ws.extent[0, 0, i, :] = o["extent"]
            ws.category[0, 0, i] = o["category"]
            ws.track_id[0, 0, i] = o["track_id"]
            ws.is_valid[0, 0, i] = 1

            if "dynamics" in o:
                ws.dynamics[0, 0, i, :] = o["dynamics"]
            if "attribute" in o:
                ws.attribute[0, 0, i] = o["attribute"]

        ws.attribute_map = CARLA_ATTRIBUTE_MAP
        ws.category_map = CARLA_CATEGORY_MAP

        # build traffic control elements
        all_tces = local_traffic_lights + local_static_signs
        tces = TrafficControlElements._create_zeros(b=1, t=1, n=len(all_tces))
        for i, tce in enumerate(all_tces):
            tces.transform[0, 0, i, :, :] = tce["world_tf_box"]
            tces.extent[0, 0, i, :] = tce["extent"]
            tces.category[0, 0, i] = tce["category"]
            tces.tce_id[0, 0, i] = tce["track_id"]
            tces.is_valid[0, 0, i] = 1
            tces.state[0, 0, i] = tce["attribute"]

        tces.state_map = CARLA_ATTRIBUTE_MAP
        tces.category_map = CARLA_CATEGORY_MAP

        return ws, tces, ego_id, timestamp, frame_id

    def extract_map(self):
        raise NotADirectoryError()

    def _extract_dynamic_objects(self, actors: dict[str, carla.Actor]):
        """Extract dynamic actors from the world."""
        result = []
        found_ego = False
        for actor in actors.values():
            class_string = actor.type_id.split(".")[0]

            # filter out non-dynamic objects
            if class_string not in ("vehicle", "walker", "static"):
                continue

            # filter out unlabeled objects
            category = get_category_from_blueprint(actor.type_id)
            attribute = get_attribute_from_blueprint(actor.type_id)
            if category is None:
                continue
            if CarlaDataProvider.open_doors[actor.id]:
                attribute = "vehicle.open_door"

            # In Carla the transform of an actor is not neccesarily the same as the center of the bounding box.
            # Rather the bounding box might be offseted by an additional transformation.
            # As defined in the worldstate, our stored transform should be the transform of the center of the bounding box
            # We account for this via the car_T_bb transform
            trans = actor.get_transform()
            pos = convert_carla_vector(trans.location)
            rot = convert_carla_rotation(trans.rotation)
            world_tf_car = TransformsOperations.get_transforms_pos_rot(pos, rot)

            # Additional transformation due to BB not being aligned with actor transform
            actor_bb = actor.bounding_box
            bb_location = convert_carla_vector(actor_bb.location)

            # rotation, extent and dynamics of static objects is broken
            # TODO: check whether this holds for small towns
            if class_string != "static":
                # normal case
                bb_rotation = convert_carla_rotation(actor_bb.rotation)
                extent = 2 * convert_carla_vector_noflip(actor_bb.extent)

                vel = np.linalg.norm(convert_carla_vector(actor.get_velocity()))
                ang_vel = actor.get_angular_velocity().z
                acc = np.linalg.norm(convert_carla_vector(actor.get_acceleration()))
            else:
                # patch
                bb_rotation = convert_carla_rotation(carla.Rotation())
                extent = CarlaWorldStateExtractor._get_extent_for_bp(actor.type_id)

                vel, ang_vel, acc = 0, 0, 0

            car_tf_box = TransformsOperations.get_transforms_pos_rot(bb_location, bb_rotation)
            world_tf_box = world_tf_car @ car_tf_box

            annotation_item = {
                "world_tf_box": world_tf_box,
                "extent": extent,
                "dynamics": np.array([vel, acc, ang_vel, 0], dtype=np.float32),
                "category": CARLA_CATEGORY_MAP[category],
                "attribute": CARLA_ATTRIBUTE_MAP[attribute],
                "track_id": self.idg.id_for_actor(actor),
            }

            # make sure to add the ego annotation at the beginning of the list
            if actor.attributes["role_name"] == "hero":
                found_ego = True
                result.insert(0, annotation_item)
            else:
                result.append(annotation_item)

        if not found_ego:
            raise ValueError("No ego vehicle found in current world state.")

        return result

    @abstractmethod
    def _extract_parked_vehicles(self):
        """Extract static (parked) vehicles from the world."""
        raise NotImplementedError()

    @abstractmethod
    def _extract_traffic_lights(self):
        """Extract stateful traffic lights from the world."""
        raise NotImplementedError()

    def _extract_static_traffic_signs(self):
        """Extract static traffic signs (they belong to the map) from the world.

        Assume: Traffic signs are modeled as environment objects and can be cached.
        TODO:
        - check whether extracting env objects instead of actors is indeed more accurate
        - check whether custom logic for special towns needs to be implemented in subclasses
        - check speed limit subclassifaction
        """

        # check for cache hit:
        if not self._cached_static_traffic_signs:
            self._cached_static_traffic_signs = []
            for eo in self.world.get_environment_objects(object_type=carla.CityObjectLabel.TrafficSigns):
                sign_type = get_static_traffic_sign_type(eo)

                # filter traffic signs which cannot be classified
                if not sign_type:
                    continue

                # filter poles (only some traffic signs have labeled poles)
                if eo.bounding_box.extent.x * 2 < eo.bounding_box.extent.z:
                    # print("Filter pole for traffic sign", eo.name)
                    continue

                pos = convert_carla_vector(eo.bounding_box.location)
                rot = convert_carla_rotation(eo.bounding_box.rotation)
                world_tf_box = TransformsOperations.get_transforms_pos_rot(pos, rot)

                v = {
                    "world_tf_box": world_tf_box,
                    "extent": 2 * convert_carla_vector_noflip(eo.bounding_box.extent),
                    "category": CARLA_CATEGORY_MAP["traffic_sign"],
                    "attribute": CARLA_ATTRIBUTE_MAP[sign_type],
                    "track_id": self.idg.id_for_int_val(eo.id),
                }
                self._cached_static_traffic_signs.append(v)

        return self._cached_static_traffic_signs

    @staticmethod
    def _get_extent_for_bp(blueprint: str):
        """Return the full-size extent for a given blueprint.

        This method is necessary since CARLA's API yields incorrect dimensions for some static assets.

        Args:
            blueprint (str): The blueprint to get the extent for.
        """
        return {
            "static.prop.warningconstruction": [1.31, 1.06, 1.86],
            "static.prop.streetbarrier": [1.21, 0.37, 1.07],
            "static.prop.trafficcone02": [0.46, 0.39, 1.18],
            "static.prop.trafficwarning": [2.37, 2.87, 3.57],
            "static.prop.trafficcone01": [0.88, 0.88, 1.13],
            "static.prop.constructioncone": [0.34, 0.34, 0.59],
            "static.prop.warningaccident": [1.31, 1.06, 1.86],
        }[blueprint]


class SmallMapWorldStateExtractor(CarlaWorldStateExtractor):
    """A special world state extractor that comes with some workarounds for the small maps
    Town01 - Town10HD..
    """

    def __init__(self, world):
        super().__init__(world=world)
        self._cached_parked_vehicles = None

        self._cached_map = None

    def extract_map(self):
        if not self._cached_map:
            self._cached_map = CarlaMapConverter.convert_map(CarlaDataProvider.get_map())
        return self._cached_map

    def _extract_parked_vehicles(self, actors_dict):
        """Extract static (parked) vehicles from the world.

        Parked vehicles are modeled as environment objects in small towns and can be cached.
        """

        # check for cache hit:
        if self._cached_parked_vehicles:
            return self._cached_parked_vehicles

        city_labels_vehicles = [
            carla.CityObjectLabel.Bicycle,
            carla.CityObjectLabel.Bus,
            carla.CityObjectLabel.Car,
            carla.CityObjectLabel.Motorcycle,
            carla.CityObjectLabel.Truck,
        ]

        result = []
        for allowed_type in city_labels_vehicles:
            # filtering objects using get_environment_objects is much faster
            for eo in self.world.get_environment_objects(object_type=allowed_type):
                # filter invalid small vehicle bounding boxes
                if eo.type in (
                    carla.CityObjectLabel.Bus,
                    carla.CityObjectLabel.Car,
                    carla.CityObjectLabel.Truck,
                ) and (eo.bounding_box.extent.z < 0.5):
                    continue

                # repair bounding boxes with negativ dimensions
                if eo.bounding_box.extent.x < 0:
                    eo.bounding_box.extent.x *= -1

                    # yes, you do have to flip the box if x is negative :D
                    eo.bounding_box.rotation.yaw += 180
                    if eo.bounding_box.rotation.yaw > 180:
                        eo.bounding_box.rotation.yaw -= 360

                pos = convert_carla_vector(eo.bounding_box.location)
                rot = convert_carla_rotation(eo.bounding_box.rotation)
                world_tf_box = TransformsOperations.get_transforms_pos_rot(pos, rot)

                category = get_category_from_city_object_label(eo.type)
                v = {
                    "world_tf_box": world_tf_box,
                    "extent": 2 * convert_carla_vector_noflip(eo.bounding_box.extent),
                    "category": CARLA_CATEGORY_MAP[category],
                    "attribute": CARLA_ATTRIBUTE_MAP["vehicle.parking"],
                    "track_id": self.idg.id_for_int_val(eo.id),
                }
                result.append(v)

        # update cache
        self._cached_parked_vehicles = result
        return result

    def _extract_traffic_lights(self, actors: dict[str, carla.Actor]):
        """Extract stateful traffic lights from the world."""

        result = []
        for tl in filter(lambda a: a.type_id == "traffic.traffic_light", actors.values()):
            tl_status = get_carla_traffic_light_state(tl.state)

            for light_box_id, light_box in enumerate(tl.get_light_boxes()):
                # extract light box size
                size = 2 * convert_carla_vector_noflip(light_box.extent)

                # filter call buttons (they are annotated as light box in CARLA)
                if size[-1] < CALL_BUTTON_Z_THRESHOLD:
                    continue

                # extract light box transform
                pos = convert_carla_vector(light_box.location)
                rot = convert_carla_rotation(light_box.rotation)
                world_tf_box = TransformsOperations.get_transforms_pos_rot(pos, rot)

                annotation_item = {
                    "world_tf_box": world_tf_box,
                    "extent": size,
                    "category": CARLA_CATEGORY_MAP["traffic_light"],
                    "attribute": CARLA_ATTRIBUTE_MAP[tl_status],
                    "track_id": self.idg.id_for_light_box(tl, light_box_id),
                }
                result.append(annotation_item)

        return result


class Town15WorldStateExtractor(SmallMapWorldStateExtractor):
    """TODO: Remove this class once the map is correctly implemented."""

    def extract_map(self):
        # TODO: implement me
        return None


class LargeMapWorldStateExtractor(CarlaWorldStateExtractor):
    """A special world state extractor that comes with some workarounds for the large maps
    Town11, Town12 and Town13.
    """

    def __init__(self, world):
        super().__init__(world=world)

        # map actor ids to a list of (stateless) traffic light annotation items
        self._cached_stateless_tl_boxes = defaultdict(list)

        # a Nx9 dimensional numpy array caching [loc, rot, ext] of each traffic light environment object
        self._env_tl_array = self._build_env_tl_array()

    def extract_map(self):
        # TODO: implement me
        return None

    def _extract_parked_vehicles(self, actors):
        """Extract static (parked) vehicles from the world."""

        result = []
        for mesh in filter(lambda a: a.type_id == "static.prop.mesh", actors.values()):
            pos = convert_carla_vector(mesh.get_transform().location)
            rot = convert_carla_rotation(mesh.get_transform().rotation)
            world_tf_actor = TransformsOperations.get_transforms_pos_rot(pos, rot)

            pos_box = convert_carla_vector(mesh.bounding_box.location)
            # we do not use the rotation of the bounding box, it is broken!
            rot_box = convert_carla_rotation(carla.Rotation())

            actor_tf_box = TransformsOperations.get_transforms_pos_rot(pos_box, rot_box)
            world_tf_box = world_tf_actor @ actor_tf_box

            # we do not use the extent of the bounding box, it is broken!
            size = self._get_size_from_mesh(mesh.attributes["mesh_path"])

            annotation_item = {
                "world_tf_box": world_tf_box,
                "extent": size,
                "category": CARLA_CATEGORY_MAP[get_category_from_mesh(mesh)],
                "attribute": CARLA_ATTRIBUTE_MAP["vehicle.parking"],
                "track_id": self.idg.id_for_int_val(mesh.id),
            }
            result.append(annotation_item)
        return result

    def _extract_traffic_lights(self, actors: dict[str, carla.Actor]):
        """Extract stateful traffic lights from the world."""

        result = []
        for tl in filter(lambda a: a.type_id == "traffic.traffic_light", actors.values()):
            if tl.is_dormant:
                # in large towns of CARLA, distant traffic lights are dormant and hence will not provide any data
                continue

            if tl.id not in self._cached_stateless_tl_boxes:
                # no cache hit: compute association of this tl's light boxes to environment objects
                self._compute_actor_env_object_association(tl)

            # list of (stateless) light box annotation items
            annotation_items = self._cached_stateless_tl_boxes[tl.id]

            # read tl state
            tl_status = get_carla_traffic_light_state(tl.state)

            # add state to each stateless annotations
            for stateless_item in annotation_items:
                stateful_item = deepcopy(stateless_item)
                stateful_item["attribute"] = CARLA_ATTRIBUTE_MAP[tl_status]
                result.append(stateful_item)

        return result

    def _build_env_tl_array(self):
        loc_rot_ext_arrays = []
        for eo in self.world.get_environment_objects(object_type=carla.CityObjectLabel.TrafficLight):
            arr = np.array(
                [
                    eo.bounding_box.location.x,
                    eo.bounding_box.location.y,
                    eo.bounding_box.location.z,
                    eo.bounding_box.rotation.roll,
                    eo.bounding_box.rotation.pitch,
                    eo.bounding_box.rotation.yaw,
                    eo.bounding_box.extent.x,
                    eo.bounding_box.extent.y,
                    eo.bounding_box.extent.z,
                ]
            )
            loc_rot_ext_arrays.append(arr)

        return np.array(loc_rot_ext_arrays)

    def _compute_actor_env_object_association(self, tl):
        for light_box_id, light_box in enumerate(tl.get_light_boxes()):
            loc = np.array([light_box.location.x, light_box.location.y, light_box.location.z])
            rot = np.array([light_box.rotation.roll, light_box.rotation.pitch, light_box.rotation.yaw])
            ext = np.array([light_box.extent.x, light_box.extent.y, light_box.extent.z])

            loc_idx = np.abs(self._env_tl_array[:, 2] - loc[2]) < 0.005  # match by world z-coordinate
            rot_idx = np.abs(self._env_tl_array[:, 3:6] - rot) < 0.005  # match by rotation parameters
            ext_idx = np.abs(self._env_tl_array[:, 6:] - ext) < 0.01  # match by extent
            candidate_idx = np.logical_and(
                loc_idx,
                np.logical_and(np.all(rot_idx, axis=-1), np.all(ext_idx, axis=-1)),
            )

            candidates = self._env_tl_array[candidate_idx, :]
            if len(candidates) < 0:
                print("Error: No corresponding env object found. Skipping.")
                continue
            if len(candidates) > 1:
                # TODO: warn about ambiguous associations
                pass
            loc_rot_ext = candidates[0, :]  # disambiguate by picking the first candidate

            pos = convert_carla_vector(carla.Location(loc_rot_ext[0], loc_rot_ext[1], loc_rot_ext[2]))
            rot = convert_carla_rotation(light_box.rotation)
            world_tf_box = TransformsOperations.get_transforms_pos_rot(pos, rot)
            size = 2 * convert_carla_vector_noflip(light_box.extent)

            # fix light boxes that are too slim by assigning a default width / length of 0.4m
            if size[0] < 0.2:
                size[0] = 0.4
            if size[1] < 0.2:
                size[1] = 0.4

            stateless_annotation = {
                "world_tf_box": world_tf_box,
                "extent": size,
                "category": CARLA_CATEGORY_MAP["traffic_light"],
                "track_id": self.idg.id_for_light_box(tl, light_box_id),
            }

            # update cache
            self._cached_stateless_tl_boxes[tl.id].append(stateless_annotation)

    def _get_size_from_mesh(self, mesh_name: str) -> np.ndarray:
        """Return the (full) size of static vehicles. All sizes are scaled by 0.9."""
        if (
            mesh_name
            == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked"
        ):
            return np.array([4.206, 1.631, 1.298])
        elif (
            mesh_name
            == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked"
        ):
            return np.array([4.829, 1.621, 1.417])
        elif mesh_name == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked":
            return np.array([4.403, 1.653, 1.341])
        elif (
            mesh_name == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked"
        ):
            return np.array([4.097, 1.887, 1.590])
        elif (
            mesh_name
            == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked"
        ):
            return np.array([4.032, 1.862, 1.834])
        elif (
            mesh_name
            == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked"
        ):
            return np.array([5.009, 1.935, 1.841])
        elif mesh_name == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked":
            return np.array([4.507, 1.693, 1.381])
        elif mesh_name == "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked":
            return np.array([4.313, 1.947, 1.339])
        else:
            raise ValueError(f"Unknown mesh: '{mesh_name}'")
