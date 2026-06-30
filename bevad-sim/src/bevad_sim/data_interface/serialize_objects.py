import dataclasses
import inspect
import io
import pickle
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import zstandard

import bevad_sim.data_interface.core_container
import bevad_sim.data_interface.episode_map
import bevad_sim.data_interface.episode_meta
import bevad_sim.data_interface.odometry
import bevad_sim.data_interface.step_meta
import bevad_sim.data_interface.tensor_observation
import bevad_sim.data_interface.world_state
from bevad_sim.data_interface.base_entity import BaseEntity


class IOUtils:
    """
    Utility class for file and directory operations, serialization, and compression.
    This class provides static methods for checking file existence, retrieving the
    home directory, creating directories, and serializing/deserializing Python objects
    using pickle and zstandard compression.

    Methods:
        file_exists(f): Check if a file exists at the given path.
        get_home(): Get the current user's home directory as a string.
        check_and_create(path): Create a directory (and parents) if it does not exist.
        write_struct(struct, fname): Serialize and compress a Python object to a file.
        read_struct(fname): Decompress and deserialize a Python object from a file.
        get_file_extension(): Get the default file extension for serialized files.
    """

    @staticmethod
    def file_exists(f):
        return Path(f).is_file()

    @staticmethod
    def get_home():
        return str(Path.home())

    @staticmethod
    def check_and_create(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_struct(struct, fname):
        binary_st = pickle.dumps(struct)
        with zstandard.open(fname, "wb") as f:
            f.write(binary_st)

    @staticmethod
    def read_struct(fname):
        with zstandard.open(fname, "rb") as f:
            content = f.read()
            struct = pickle.loads(content)
            return struct

    @staticmethod
    def get_file_extension():
        return "zstd"


def get_serializable_objects():
    """
    Returns a set of fully qualified names of serializable object types.
    The set includes common built-in types, NumPy types, and specific
    data structure types used within the bevad_sim project. These types
    are considered serializable for the purposes of data interchange
    or persistence.

    Returns:
        set: A set of strings, each representing the fully qualified name
            of a serializable object type.
    """
    res = set()
    res.add("numpy.ndarray")
    res.add("numpy.float64")
    res.add("numpy.int32")
    res.add("int")
    res.add("str")
    res.add("float")
    res.add("bool")
    res.add("NoneType")

    res.add("bevad_sim.common.data_structures.data_types.BoundaryType")
    res.add("bevad_sim.common.data_structures.data_types.DynamicAgentType")
    res.add("bevad_sim.common.data_structures.data_types.HighLevelCommands")
    res.add("bevad_sim.common.data_structures.data_types.LaneType")
    res.add("bevad_sim.common.data_structures.data_types.SensorTypes")
    res.add("bevad_sim.common.data_structures.data_types.StaticObstacleType")
    res.add("bevad_sim.common.data_structures.data_types.TrafficControlElementType")
    res.add("bevad_sim.common.data_structures.data_types.TrafficLightState")

    return res


def fullname(o):
    """
    Returns the fully qualified name of an object's class.
    If the object's class is a built-in type, only the class name is returned.
    Otherwise, the result is in the format 'module.ClassName'.

    Args:
        o (object): The object whose class name is to be retrieved.

    Returns:
        str: The fully qualified class name of the object.
    """

    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def is_serializable(obj):
    """
    Determines whether the given object is serializable based on its type.

    Args:
        obj: The object to check for serializability.

    Returns:
        bool: True if the object's type is in the set of serializable objects, False otherwise.
    """
    obj_type = fullname(obj)

    obj_set = get_serializable_objects()

    if obj_type in obj_set:
        return True

    return False


def is_serializable_type(obj_type):
    """
    Determines whether the given object type is serializable.

    Args:
        obj_type: The type of the object to check for serializability.

    Returns:
        bool: True if the object type is serializable, False otherwise.
    """
    obj_set = get_serializable_objects()

    if obj_type in obj_set:
        return True

    return False


def is_default_serializable(obj):
    """
    Determines if the given object is of a default serializable type.
    This function checks whether the object's type is one of the basic built-in types
    that are typically serializable by default, such as int, str, float, bool, or NoneType.

    Args:
        obj: The object to check for default serializability.

    Returns:
        bool: True if the object is of a default serializable type, False otherwise.
    """

    obj_type = fullname(obj)

    if obj_type in ("int", "str", "float", "bool", "NoneType"):
        return True

    return False


def is_container(obj):
    """
    Determines if the given object is a container type (list, tuple, or dict).

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a list, tuple, or dict; False otherwise.
    """
    obj_type = fullname(obj)

    if obj_type in ("list", "tuple", "dict"):
        return True
    return False


class ObjectSerializer:
    """Serializes Python objects for storage or transmission.
    This class provides methods to serialize various Python objects, including
    NumPy arrays, primitive types, custom data classes, and containers such as
    lists, tuples, and dictionaries. It supports recursive serialization of
    nested objects and handles special cases for certain data types.

    Attributes:
        output_folder (str): The directory where serialized objects may be stored.
        config (Any): Configuration object used to control serialization behavior.

    Methods:
        serialize_obj(obj):
            Serializes a single object based on its type.
        serialize_container(obj):
            Serializes container objects (list, tuple, dict) recursively.
        serialize_obj_rec_(obj):
            Recursively serializes an object, including its attributes, using the current approach.
    """

    def __init__(self, output_folder, config=None):
        self.output_folder = output_folder
        self.config = config

    def serialize_obj(self, obj):
        """
        Serializes a given object into a format suitable for storage or transmission.
        The serialization method depends on the object's type:
          - For numpy.ndarray: serializes as a tuple containing the dtype, shape, and raw bytes.
          - For primitive types (int, float, str, bool, NoneType, numpy.float64, numpy.int32): uses pickle serialization.
          - For objects from 'bevad_sim.data_interface.data_types': serializes as the integer representation using pickle.
          - For objects with a 'serialize_data_class' method: calls that method and returns its result.
          - Returns None if the object type is not supported.

        Args:
            obj (Any): The object to serialize.

        Returns:
            Any: The serialized representation of the object, or None if the type is unsupported.

        Raises:
            Exception: Propagates exceptions raised during serialization.
        """

        obj_type = fullname(obj)
        if obj_type == "numpy.ndarray":
            # Not using pickle or numpy's .tobytes() because these are not version-stable
            f = io.BytesIO()
            np.save(f, obj)
            return "numpy.ndarray_bytesio", f.getvalue()

        if obj_type in [
            "numpy.float64",
            "numpy.int64",
            "numpy.int32",
            "int",
            "str",
            "float",
            "bool",
            "NoneType",
        ]:
            return pickle.dumps(obj)

        # TODO(jujorda): check for int enum
        if "bevad_sim.data_interface.data_types" in obj_type:
            return pickle.dumps(int(obj))

        if hasattr(obj, "serialize_data_class"):
            return obj_type, obj.serialize_data_class(self, self.output_folder)

        ### TODO: Proper implement tensor_observation read/write
        # if obj_type == "bevad_sim.common.data_structures.sensor_observation.MeasurementIOInfo":
        #    return obsio.write_measurement(self, obj)

        return None

    def serialize_container(self, obj):
        """
        Serializes a container object (list, tuple, or dict) by recursively serializing its elements.

        Args:
            obj (Any): The container object to serialize. Supported types are list, tuple, and dict.

        Returns:
            Any: A serialized version of the input container, with all elements recursively serialized.
                  - For lists: returns a list of serialized elements.
                  - For tuples: returns a tuple of serialized elements.
                  - For dicts: returns a dict with serialized values.

        Raises:
            TypeError: If the input object is not a supported container type.
        """

        obj_type = fullname(obj)
        if obj_type == "list":
            lres = []
            for i, lobj in enumerate(obj):
                lres.append(self.serialize_obj_rec_(lobj))
            return lres

        if obj_type == "tuple":
            lres = []
            for i, lobj in enumerate(obj):
                lres.append(self.serialize_obj_rec_(lobj))
            return tuple(lres)

        if obj_type == "dict":
            dres = {}
            for dkey in obj:
                dres[dkey] = self.serialize_obj_rec_(obj[dkey])
            return dres

    def serialize_obj_rec_(self, obj, ignore_root_fields=set()):
        """
        Recursively serializes an object, handling containers, serializable objects, and custom objects.
        This method checks if the given object is a container or a serializable object and delegates
        serialization accordingly. For custom objects, it recursively serializes all attributes
        if they derive from BaseEntity.

        Args:
            obj (Any): The object to serialize.

        Returns:
            Union[Any, Tuple[str, dict]]: The serialized representation of the object. If the object is a
            container or serializable, returns the result of their respective serialization methods.
            Otherwise, returns a tuple containing the object's fully qualified name and a dictionary of
            its serialized public attributes which derive from BaseEntity.
        """
        if is_container(obj):
            return self.serialize_container(obj)

        obj_name = fullname(obj)

        tres = self.serialize_obj(obj)
        if tres is not None:
            return tres

        obj_dict = obj.__dict__

        res = {}
        for okey in obj_dict:
            if okey[0] == "_" or okey in ignore_root_fields:
                continue
            res[okey] = self.serialize_obj_rec_(obj_dict[okey])
        return obj_name, res


def get_class_from_type_direct(obj_type):
    """
    Retrieves a class object from a fully qualified class name string.

    Args:
        obj_type (str): The fully qualified class name (e.g., 'package.module.ClassName').

    Returns:
        type or None: The class object if found, otherwise None.
    """
    try:
        module_path, class_name = obj_type.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None


def get_class_from_type(obj_type, class_path_hints: Optional[Dict] = None):
    """
    Returns the class object corresponding to the given object type string.
    This function maps a string representing the type of an object to its corresponding
    class within the bevad_sim.data_interface module. The mapping is based on the last token
    of the dot-separated `obj_type` string. If the type is not recognized, the function returns None.

    Args:
        obj_type (str): The dot-separated string representing the object type.
        class_path_hints (Dict): Dict with lookup table from object name to class paths, path can either be class directly of string path.

    Returns:
        type or None: The class object corresponding to the given type, or None if not found.

    Note:
        This function currently supports only a predefined set of core classes and is not generic.
        Refactoring may be required to support custom classes.
    """

    tokens = obj_type.split(".")

    # Class path hints override the default class paths
    if class_path_hints is not None and tokens[-1] in class_path_hints:
        class_path = class_path_hints[tokens[-1]]
        if isinstance(class_path, str):
            module_path, class_name = class_path.rsplit(".", 1)
            module = import_module(module_path)
            return getattr(module, class_name)

        if inspect.isclass(class_path):
            return class_path

        raise ValueError(f" Class path for class name: {tokens[-1]} is not class or string:  {class_path}")

    if tokens[-1] in ("CoreContainer", "Episode"):
        return bevad_sim.data_interface.core_container.CoreContainer
    if tokens[-1] == "WorldState":
        return bevad_sim.data_interface.world_state.WorldState
    if tokens[-1] == "TrafficControlElements":
        return bevad_sim.data_interface.tce.TrafficControlElements
    if tokens[-1] == "TrafficControlElementInfo":
        return bevad_sim.data_interface.episode_map.TrafficControlElementInfo
    if tokens[-1] == "MapContainer":
        return bevad_sim.data_interface.episode_map.MapContainer
    if tokens[-1] == "EpisodeMap":
        return bevad_sim.data_interface.episode_map.EpisodeMap
    if tokens[-1] == "LaneSegmentInfo":
        return bevad_sim.data_interface.episode_map.LaneSegmentInfo
    if tokens[-1] == "CrosswalkInfo":
        return bevad_sim.data_interface.episode_map.CrosswalkInfo
    if tokens[-1] == "EpisodeMeta":
        return bevad_sim.data_interface.episode_meta.EpisodeMeta
    if tokens[-1] == "StepMeta":
        return bevad_sim.data_interface.step_meta.StepMeta
    if tokens[-1] == "Action":
        return bevad_sim.data_interface.action.Action
    if tokens[-1] == "Odometry":
        return bevad_sim.data_interface.odometry.Odometry
    if tokens[-1] == "RoutingInformation":
        return bevad_sim.data_interface.routing_information.RoutingInformation
    if tokens[-1] == "TensorObservation":
        return bevad_sim.data_interface.tensor_observation.TensorObservation
    if tokens[-1] == "CameraObservation":
        return bevad_sim.data_interface.tensor_observation.CameraObservation
    if tokens[-1] == "LidarObservation":
        return bevad_sim.data_interface.tensor_observation.LidarObservation
    if tokens[-1] == "RadarObservation":
        return bevad_sim.data_interface.tensor_observation.RadarObservation

    return None


class ObjectDeSerializer:
    """
    A class for deserializing various Python objects from serialized representations.
    This class provides methods to reconstruct Python objects, including NumPy arrays,
    primitive types, and custom classes, from their serialized forms. It supports
    recursive deserialization of nested containers and custom objects, leveraging
    class methods and dataclass constructors where appropriate.

    Attributes:
        input_folder (str): The folder path used for loading additional data required by some objects.
        load_payload (bool): Bool flag for specifying if payload, i.e. sensor data, should be loaded upfront or not.
        class_path_hints (Dict[class or str]): If not None, contains paths to classes for specified objects. Can either be a class directly or a path as str.

    Methods:
        de_serialize_obj_old(obj, obj_type):
            Deserializes an object based on its type using an older serialization format.
        de_serialize_obj(obj):
            Deserializes an object from a tuple containing its type and serialized data.
        de_serialize_generic(byte_data):
            Deserializes a generic object from raw byte data.
        de_serialize_container(obj):
            Recursively deserializes container objects such as lists, tuples, and dictionaries.
        obj_type_to_class(obj_type):
            Resolves a string type name to the corresponding Python class.
        de_serialize_obj_rec_(obj):
            Recursively deserializes objects, handling custom classes, containers, and primitive types.
    """

    def __init__(self, input_folder, load_payload: bool, class_path_hints: Optional[Dict] = None):
        self.input_folder = input_folder
        self.load_payload = load_payload
        self.class_path_hints = class_path_hints

    def de_serialize_obj(self, obj):
        """
        Deserializes an object from its serialized representation.
        Supports deserialization of numpy arrays, primitive types (int, str, float, NoneType, bool),
        and custom objects with a `load_data` method. For numpy arrays, reconstructs the array from
        its dtype, shape, and buffer. For custom objects, recursively deserializes their dictionary
        representation and loads additional data if required.

        Args:
            obj (Any): The serialized object to deserialize.

        Returns:
            Tuple[Any, bool]: A tuple containing the deserialized object and a boolean indicating
                whether deserialization was successful.
        """

        obj_type = obj[0]
        if obj_type == "numpy.ndarray_bytesio":
            return np.load(io.BytesIO(obj[1])), True

        ### This is for backwards compatibility with bevad_sim up to version 0.4. Remove once test episodes are updated and old data is not required
        # anymore.
        if obj_type == "numpy.ndarray":
            version_number = [int(x) for x in np.version.version.split(".")]
            right_numpy_version = version_number[0] == 1 or (
                version_number[0] == 2 and version_number[1] <= 2 and version_number[2] <= 5
            )
            assert right_numpy_version, "Wrong numpy version, only < 2.25 is supported with np.frombuffer"
            return (np.frombuffer(obj[1][2], np.dtype(obj[1][0])).reshape(obj[1][1]).copy(), True)

        if obj_type == "int":
            return pickle.loads(obj), True

        if obj_type == "str":
            return pickle.loads(obj), True

        if obj_type == "float":
            return pickle.loads(obj), True

        if obj_type == "NoneType":
            return pickle.loads(obj), True

        if obj_type == "bool":
            return pickle.loads(obj), True

        return None, False

    def de_serialize_generic(self, byte_data):
        """
        Deserializes a generic Python object from a bytes-like object.

        Args:
            byte_data (bytes): The byte stream to deserialize.

        Returns:
            Any: The deserialized Python object.
        """

        return pickle.loads(byte_data)

    def de_serialize_container(self, obj):
        """
        Recursively deserializes container objects (list, tuple, dict) by processing their elements.

        Args:
            obj (Any): The container object to be deserialized. Supported types are list, tuple, and dict.

        Returns:
            Any: A new container of the same type with all elements recursively deserialized.
        """

        obj_type = fullname(obj)
        if obj_type == "list":
            lres = []
            for i, lobj in enumerate(obj):
                lres.append(self.de_serialize_obj_rec_(lobj))
            return lres

        if obj_type == "tuple":
            lres = []
            for i, lobj in enumerate(obj):
                lres.append(self.de_serialize_obj_rec_(lobj))
            return tuple(lres)

        if obj_type == "dict":
            dres = {}
            for dkey in obj:
                dres[dkey] = self.de_serialize_obj_rec_(obj[dkey])
            return dres

    def obj_type_to_class(self, obj_type):
        """
        Resolves a string representing an object type to its corresponding class.

        Args:
            obj_type (str): The fully qualified type name of the object.

        Returns:
            type or None: The class corresponding to the given object type, or None if not found or invalid input.

        Raises:
            ValueError: If the object type cannot be resolved to a class.
        """

        if type(obj_type).__name__ != "str":
            return None

        obj_class = get_class_from_type(obj_type, self.class_path_hints)
        if obj_class is not None:
            return obj_class

        ### TODO: Only allow this for classes derived from base entity?
        obj_class = get_class_from_type_direct(obj_type)
        if obj_class is not None:
            return obj_class

        raise ValueError

    def de_serialize_obj_rec_(self, obj):
        """
        Recursively deserializes an object from its serialized representation.

        Args:
            obj: The serialized object, which can be a tuple, container, bytes, or other supported types.

        Returns:
            The deserialized Python object.

        Raises:
            ValueError: If the object type is unknown or unsupported.
        """

        if type(obj).__name__ == "tuple" and len(obj) == 2 and type(obj[0]).__name__ == "str":
            tobj = self.de_serialize_obj(obj)
            if tobj[1]:
                return tobj[0]

        if type(obj).__name__ == "tuple" and len(obj) == 2:
            obj_class = self.obj_type_to_class(obj[0])
            obj_dict = obj[1]

            if obj_class is not None:
                # to deserialize a object type, it must either have a create_empty classmethod or a default constructor
                if hasattr(obj_class, "create_empty"):
                    res_obj = obj_class.create_empty()
                elif dataclasses.is_dataclass(obj_class):  # Is this required?
                    res_obj = obj_class(
                        **{k: self.de_serialize_obj_rec_(obj_dict[k]) for k in obj_class.__dataclass_fields__.keys()}
                    )
                else:
                    # call the default constructor
                    res_obj = obj_class()

                for key in obj_dict:
                    res_obj.__dict__[key] = self.de_serialize_obj_rec_(obj_dict[key])

                if hasattr(obj_class, "load_data"):
                    if self.load_payload:
                        res_obj.load_data(self.input_folder)
                    else:
                        res_obj.set_base_data_folder(self.input_folder)

                return res_obj

        if is_container(obj):
            return self.de_serialize_container(obj)

        if type(obj).__name__ == "bytes":
            return pickle.loads(obj)

        print(f"unknown object type: {type(obj).__name__}")
        raise ValueError
