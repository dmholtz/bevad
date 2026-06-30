from enum import Enum, IntEnum


class ValidGTS(Enum):
    ASSOCIATION = "association"
    BUFFER_ZONE = "buffer_zone"
    CALIBRATION_ESTIMATE = "calibration_estimate"
    CROSSWALK = "crosswalk"
    DYNAMIC_OCCUPANCY = "dynamic_occupancy"
    EGOMOTION_ESTIMATE = "egomotion_estimate"
    GORE_AREA = "gore_area"
    INTERSECTION_AREA = "intersection_area"
    LABEL_CLASS = "label_class"
    LANE_LINE = "lane_line"
    LANE = "lane"
    MADS_TRACE = "mads_trace"
    NAV_MAP = "nav_map"
    OBSTACLE = "obstacle"
    POLE = "pole"
    ROAD_BOUNDARY = "road_boundary"
    ROAD_ISLAND = "road_island"
    ROAD_MARKING = "road_marking"
    STATIC_FREESPACE = "static_freespace"
    STATIC_HAZARD = "static_hazard"
    STATIC_OCCUPANCY = "static_occupancy"
    TRAFFIC_LIGHT = "traffic_light"
    TRAFFIC_SIGN = "traffic_sign"
    WAIT_LINE = "wait_line"
    WEATHER_CONDITION = "weather_condition"


class DynamicAgentType(IntEnum):
    """Enumeration of dynamic agent types in autonomous driving scenarios."""

    UNKNOWN = 1

    VEHICLE = 2
    LARGE_VEHICLE = 3

    CYCLIST = 4
    MOTORCYCLIST = 5

    PEDESTRIAN = 6
    ANIMAL = 7

    BACKGROUND = 8
    STATIC = 9
    CONSTRUCTION = 10
    BARRIER = 11
    TRAFFIC_CONE = 12


class StaticObstacleType(IntEnum):
    """Enumeration of static obstacle types in the driving environment."""

    UNKNOWN = 1
    SOLID = 2
    VEGETATION = 3


class HighLevelCommands(IntEnum):
    """Enumeration of high-level driving commands for route planning."""

    Void = (0,)
    Left = (1,)
    Right = (2,)
    Straight = (3,)
    LaneFollow = (4,)
    ChangeLaneLeft = (5,)
    ChangeLaneRight = (6,)
    RoadEnd = 7


class BoundaryType(IntEnum):
    """Enumeration of road lane boundary and marking types."""

    UNKNOWN = 1
    NONE = 2

    SOLID_WHITE = 3
    SOLID_YELLOW = 4
    SOLID_BLUE = 5

    DASHED_WHITE = 6
    DASHED_YELLOW = 7

    DOUBLE_SOLID_YELLOW = 8
    DOUBLE_SOLID_WHITE = 9

    DOUBLE_DASH_YELLOW = 10
    DOUBLE_DASH_WHITE = 11

    DASH_SOLID_WHITE = 12
    DASH_SOLID_YELLOW = 13
    SOLID_DASH_WHITE = 14
    SOLID_DASH_YELLOW = 15

    GRASS = 16
    PHYSICAL = 17
    VIRTUAL = 18


class DrivingDirection(IntEnum):
    """Enumeration of driving direction."""

    UNKNOWN = 1
    FORWARD = 2
    BACKWARD = 3
    BIDIRECTIONAL = 4
    NOT_DRIVABLE = 5


class LaneType(IntEnum):
    """Enumeration of lane types and designated uses."""

    UNKNOWN = 1
    NORMAL = 2
    BUS = 3
    BICYCLE = 4
    SIDEWALK = 5


class LaneGeometryType(IntEnum):
    """Enumeration of lane geometry element types."""

    NONE = 0
    UNKNOWN = 1
    LEFT_BOUNDARY = 2
    RIGHT_BOUNDARY = 3
    DRIVEABLE_SPACE = 4


class SensorTypes(IntEnum):
    """Enumeration of sensor types for perception."""

    CAMERA_RGB = 1
    CAMERA_DEPTH = 2
    CAMERA_SEMANTIC = 3
    LIDAR = 4
    RADAR = 5
    COLLISION = 6


class TrafficControlElementType(IntEnum):
    """Enumeration of traffic control element types."""

    UNKNOWN = 1
    TRAFFIC_LIGHT = 2
    BARRIER = 3
    STOP_SIGN = 4
    PRIORITY_SIGN = 5
    YIELD_SIGN = 6
    FOURWAYSTOP_SIGN = 7
    TURNSTOP_SIGN = 8
    # For all stops line types, the ENUM is named as stop_line_ + <category> + _ + <intersection_subtype>
    STOP_LINE_UNKNOWN_NOT_APPLICABLE = 9
    STOP_LINE_UNKNOWN_EXIT = 10
    STOP_LINE_UNKNOWN_ENTRY = 11
    STOP_LINE_STOP_NOT_APPLICABLE = 12
    STOP_LINE_STOP_CROSSWALK_ENTRY = 13
    STOP_LINE_STOP_ENTRY = 14
    STOP_LINE_YIELD_NOT_APPLICABLE = 15
    STOP_LINE_YIELD_ENTRY = 16


class TrafficLightState(IntEnum):
    """Enumeration of traffic light states and colors."""

    UNKNOWN = 1
    OFF = 2
    GREEN = 3
    YELLOW = 4
    RED = 5
