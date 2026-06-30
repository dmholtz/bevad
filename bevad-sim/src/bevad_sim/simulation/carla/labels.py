from __future__ import annotations

import re

import carla


def get_category_from_blueprint(blueprint: str):
    """Return the category string for a blueprint string."""
    if blueprint in [
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
    ]:
        return "bicycle"
    elif blueprint in [
        "vehicle.mitsubishi.fusorosa",
    ]:
        return "bus"
    elif blueprint in [
        "vehicle.audi.a2",
        "vehicle.audi.etron",
        "vehicle.audi.tt",
        "vehicle.bmw.grandtourer",
        "vehicle.chevrolet.impala",
        "vehicle.citroen.c3",
        "vehicle.dodge.charger_2020",
        "vehicle.dodge.charger_police",
        "vehicle.dodge.charger_police_2020",
        "vehicle.ford.crown",
        "vehicle.ford.mustang",
        "vehicle.jeep.wrangler_rubicon",
        "vehicle.lincoln.mkz_2017",
        "vehicle.lincoln.mkz_2020",
        "vehicle.mercedes.coupe",
        "vehicle.mercedes.coupe_2020",
        "vehicle.mercedes.sprinter",
        "vehicle.micro.microlino",
        "vehicle.mini.cooper_s",
        "vehicle.mini.cooper_s_2021",
        "vehicle.nissan.micra",
        "vehicle.nissan.patrol",
        "vehicle.nissan.patrol_2021",
        "vehicle.seat.leon",
        "vehicle.tesla.cybertruck",
        "vehicle.tesla.model3",
        "vehicle.toyota.prius",
        "vehicle.volkswagen.t2",
        "vehicle.volkswagen.t2_2021",
    ]:
        return "car"
    elif blueprint in [
        "vehicle.ford.ambulance",
    ]:
        return "ambulance"
    elif blueprint in [
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.vespa.zx125",
        "vehicle.yamaha.yzf",
    ]:
        return "motorcycle"
    elif blueprint in [
        "vehicle.carlamotors.carlacola",
        "vehicle.carlamotors.european_hgv",
        "vehicle.carlamotors.firetruck",
    ]:
        return "truck"
    elif blueprint in [
        "static.prop.constructioncone",
        "static.prop.streetbarrier",
        "static.prop.trafficcone01",
        "static.prop.trafficcone02",
        "static.prop.trafficwarning",  # trailer with warning
        "static.prop.warningaccident",  # moveable traffic sign
        "static.prop.warningconstruction",  # moveable traffic sign
    ]:
        return "roadway_item"
    elif "walker" in blueprint:
        return "pedestrian"
    return None


def get_attribute_from_blueprint(blueprint: str):
    """Returns the attribute for a blueprint."""
    if blueprint in [
        "vehicle.dodge.charger_police",
        "vehicle.dodge.charger_police_2020",
        "vehicle.carlamotors.firetruck",
        "vehicle.ford.ambulance",
    ]:
        return "vehicle.emergency"
    # default is no attribute
    return ""


def get_category_from_city_object_label(label: carla.CityObjectLabel):
    """Return the category string for a carla.CityObjectLabel instance."""
    if label in (carla.CityObjectLabel.Car,):
        return "car"
    elif label in (carla.CityObjectLabel.Bicycle,):
        return "bicycle"
    elif label in (carla.CityObjectLabel.Motorcycle,):
        return "motorcycle"
    elif label in (carla.CityObjectLabel.Bus,):
        # TODO: check if this is correct, i.e., ensure that vans are not labeled as bus
        return "bus"
    elif label in (carla.CityObjectLabel.Truck,):
        return "truck"
    else:
        raise ValueError(f"Unsupported label '{str(label)}'")


def get_category_from_mesh(mesh_actor):
    """Return the category string for a mesh actor of a static vehicle in large maps."""
    return "car"  # so far, all parked vehicles in large towns seem to be cars


def get_carla_traffic_light_state(tl_state: carla.TrafficLightState):
    """"""
    if tl_state == carla.TrafficLightState.Red:
        return "traffic_light_state.red"
    elif tl_state == carla.TrafficLightState.Yellow:
        return "traffic_light_state.yellow"
    elif tl_state == carla.TrafficLightState.Green:
        return "traffic_light_state.green"
    elif tl_state == carla.TrafficLightState.Off:
        return "traffic_light_state.off"
    elif tl_state == carla.TrafficLightState.Unknown:
        return "traffic_light_state.unknown"
    else:
        raise ValueError(f"Unknown TrafficLightState: {tl_state}")


def get_static_traffic_sign_type(eo: carla.EnvironmentObject) -> str | None:
    """
    Determine the type of a static CARLA traffic sign given an EnvironmentObject.

    Static traffic signs belong to the world and cannot be spawned by the API.
    """

    # basic signs: animal, interchange, lane_reduce, no_turn, one_way, stop, yield
    if "animal" in eo.name.lower():
        return "traffic_sign.animal"
    if "interchange" in eo.name.lower():
        return "traffic_sign.interchange"
    if "lanereduc" in eo.name.lower():
        return "traffic_sign.lane_reduce"
    if "noturns" in eo.name.lower():
        return "traffic_sign.no_turn"
    if "oneway" in eo.name.lower():
        return "traffic_sign.one_way"
    if "stop" in eo.name.lower():
        return "traffic_sign.stop"
    if "yield" in eo.name.lower():
        return "traffic_sign.yield"

    # speed limits: there are numerous naming patterns

    # used in Town01 and Town02
    pattern = r"SpeedLimiter(\d+)"
    match = re.search(pattern, eo.name)
    if match:
        val = int(match.group(1))
        if 30 <= val < 60:
            speed_limit = 30
        elif 60 <= val < 90:
            speed_limit = 60
        elif 90 <= val:
            speed_limit = 90
        return f"traffic_sign.speed_limit.{speed_limit}"

    # used in Town15, Town11
    pattern = r"SpeedLimit(\d+)_C"
    match = re.search(pattern, eo.name)
    if match:
        return f"traffic_sign.speed_limit.{match.group(1)}"

    # used in Town12, Town13
    pattern = r"SpeedLimit(\d+)_(\d+)_C"
    match = re.search(pattern, eo.name)
    if match:
        return f"traffic_sign.speed_limit.{match.group(1)}"

    # ambiguous pattern, used in all towns
    pattern = r"SpeedLimit(\d+)"
    match = re.search(pattern, eo.name)
    if match:
        return f"traffic_sign.speed_limit.unclassified"

    # cannot determine this traffic sign type
    return None


CARLA_CATEGORY_MAP = {
    "ambulance": 1,
    "bicycle": 2,
    "bus": 3,
    "car": 4,
    "construction": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "roadway_item": 8,
    "traffic_light": 9,
    "traffic_sign": 10,
    "truck": 11,
    # TODO: any else?
}

CARLA_ATTRIBUTE_MAP = {
    # no attribute
    "": 0,
    # dynamic object attributes
    "vehicle.parking": 1,
    "vehicle.driving": 2,
    "vehicle.emergency": 3,
    "vehicle.open_door": 4,
    # traffic light attributes
    "traffic_light_state.red": 100,
    "traffic_light_state.yellow": 101,
    "traffic_light_state.green": 102,
    "traffic_light_state.off": 103,
    "traffic_light_state.unknown": 104,
    # speed limits
    "traffic_sign.speed_limit.unclassified": 200,
    "traffic_sign.speed_limit.20": 220,
    "traffic_sign.speed_limit.25": 225,
    "traffic_sign.speed_limit.30": 230,
    "traffic_sign.speed_limit.50": 250,
    "traffic_sign.speed_limit.55": 255,
    "traffic_sign.speed_limit.60": 260,
    "traffic_sign.speed_limit.75": 275,
    "traffic_sign.speed_limit.90": 290,
    # generic traffic signs
    "traffic_sign.animal": 400,
    "traffic_sign.interchange": 410,
    "traffic_sign.lane_reduce": 420,
    "traffic_sign.no_turn": 430,
    "traffic_sign.one_way": 440,
    "traffic_sign.stop": 450,
    "traffic_sign.yield": 460,
    "traffic_sign.warning.accident": 471,
    "traffic_sign.warning.construction": 472,
    # TODO: add remaining
    # TODO: any else?
}

DEFAULT_WEATHERS = [
    ("ClearNight", carla.WeatherParameters.ClearNight),
    ("ClearNoon", carla.WeatherParameters.ClearNoon),
    ("ClearSunset", carla.WeatherParameters.ClearSunset),
    ("CloudyNight", carla.WeatherParameters.CloudyNight),
    ("CloudyNoon", carla.WeatherParameters.CloudyNoon),
    ("CloudySunset", carla.WeatherParameters.CloudySunset),
    ("DustStorm", carla.WeatherParameters.DustStorm),
    ("HardRainNight", carla.WeatherParameters.HardRainNight),
    ("HardRainNoon", carla.WeatherParameters.HardRainNoon),
    ("HardRainSunset", carla.WeatherParameters.HardRainSunset),
    ("MidRainSunset", carla.WeatherParameters.MidRainSunset),
    ("MidRainyNight", carla.WeatherParameters.MidRainyNight),
    ("MidRainyNoon", carla.WeatherParameters.MidRainyNoon),
    ("SoftRainNight", carla.WeatherParameters.SoftRainNight),
    ("SoftRainNoon", carla.WeatherParameters.SoftRainNoon),
    ("SoftRainSunset", carla.WeatherParameters.SoftRainSunset),
    ("WetCloudyNight", carla.WeatherParameters.WetCloudyNight),
    ("WetCloudyNoon", carla.WeatherParameters.WetCloudyNoon),
    ("WetCloudySunset", carla.WeatherParameters.WetCloudySunset),
    ("WetNight", carla.WeatherParameters.WetNight),
    ("WetNoon", carla.WeatherParameters.WetNoon),
    ("WetSunset", carla.WeatherParameters.WetSunset),
]
