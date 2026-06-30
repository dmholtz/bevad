import os
import sqlite3
import xml.etree.ElementTree as ET

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from tqdm import tqdm

Ability = {
    "Overtaking": [
        "Accident",
        "AccidentTwoWays",
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "HazardAtSideLaneTwoWays",
        "HazardAtSideLane",
        "ParkedObstacleTwoWays",
        "ParkedObstacle",
        "VehicleOpensDoorTwoWays",
    ],
    "Merging": [
        "CrossingBicycleFlow",
        "EnterActorFlow",
        "HighwayExit",
        "InterurbanActorFlow",
        "HighwayCutIn",
        "InterurbanAdvancedActorFlow",
        "MergerIntoSlowTrafficV2",
        "MergerIntoSlowTraffic",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow",
        "ParkingExit",
        "SequentialLaneChange",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
    ],
    "Emergency_Brake": [
        "BlockedIntersection",
        "DynamicObjectCrossing",
        "HardBreakRoute",
        "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight",
        "ParkingCutIn",
        "PedestrianCrossing",
        "ParkingCrossingPedestrian",
        "StaticCutIn",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "ControlLoss",
    ],
    "Give_Way": ["InvadingTurn", "YieldToEmergencyVehicle"],
    "Traffic_Signs": [
        "BlockedIntersection",
        "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight",
        "PedestrianCrossing",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "EnterActorFlow",
        "CrossingBicycleFlow",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow",
        "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight",
        "PedestrianCrossing",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
        "T_Junction",
        "VanillaNonSignalizedTurn",
        "VanillaSignalizedTurnEncounterGreenLight",
        "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurnEncounterStopsign",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
    ],
}


def get_infraction_status(record):
    for infraction, value in record["infractions"].items():
        if infraction == "min_speed_infractions":
            continue
        elif len(value) > 0:
            return True
    return False


def update_Ability(scenario_name, Ability_Statistic, status):
    for ability, scenarios in Ability.items():
        if scenario_name in scenarios:
            Ability_Statistic[ability][1] += 1
            if status:
                Ability_Statistic[ability][0] += 1
    pass


def update_Success(scenario_name, Success_Statistic, status):
    if scenario_name not in Success_Statistic:
        if status:
            Success_Statistic[scenario_name] = [1, 1]
        else:
            Success_Statistic[scenario_name] = [0, 1]
    else:
        Success_Statistic[scenario_name][1] += 1
        if status:
            Success_Statistic[scenario_name][0] += 1
    pass


def load_route(route_file):
    tree = ET.parse(route_file)
    root = tree.getroot().find("route")
    return root


def get_position(xml_route):
    waypoints_elem = xml_route.find("waypoints")
    keypoints = waypoints_elem.findall("position")
    return [
        carla.Location(float(pos.get("x")), float(pos.get("y")), float(pos.get("z")))
        for pos in keypoints
    ]


def get_route_result(records, route_id):
    for record in records:
        record_route_id = record["route_id"].split("_")[1]
        if route_id == record_route_id:
            return record
    return None


def get_waypoint_route(locs, grp):
    route = []
    for i in range(len(locs) - 1):
        loc = locs[i]
        loc_next = locs[i + 1]
        interpolated_trace = grp.trace_route(loc, loc_next)
        for wp, _ in interpolated_trace:
            route.append(wp)
    return route


def compute_ability_benchmark(xml_folder, result_db_file):
    Ability_Statistic = {}
    for key in Ability:
        Ability_Statistic[key] = [0, 0.0]
    Success_Statistic = {}

    # load results database
    result_db = sqlite3.connect(result_db_file)
    cursor = result_db.cursor()

    # connect to CARLA to inspect routes
    client = carla.Client("localhost")
    client.set_timeout(300)
    current_town = None

    res = cursor.execute(
        "SELECT id, town, scenario, route_completion, success_rate, num_stop, num_red_lights FROM episode ORDER BY town"
    ).fetchall()
    for (
        episode_id,
        town,
        scenario_name,
        route_completion,
        record_success_status,
        num_stop,
        num_red_lights,
    ) in tqdm(res):
        route_id = episode_id.split("-")[0]

        if current_town != town:
            current_town = town
            world = client.load_world(current_town)
            world.tick()
            carla_map = world.get_map()

        update_Ability(scenario_name, Ability_Statistic, record_success_status)
        update_Success(scenario_name, Success_Statistic, record_success_status)

        if scenario_name in Ability["Traffic_Signs"]:
            route_file = os.path.join(xml_folder, f"{route_id}.xml")
            route = load_route(route_file)

            grp = GlobalRoutePlanner(carla_map, 1.0)
            location_list = get_position(route)
            waypoint_route = get_waypoint_route(location_list, grp)
            count = 0
            for wp in waypoint_route:
                count += 1
                if wp.is_junction:
                    break
            if not wp.is_junction:
                raise RuntimeError("This route does not contain any junction-waypoint!")
            # +8 to ensure the ego pass the trigger volume
            junction_completion = float(count + 8) / float(len(waypoint_route))
            record_completion = route_completion / 100.0
            stop_infraction = None if num_stop < 1 else num_stop
            red_light_infraction = None if num_red_lights < 1 else num_red_lights
            if (
                record_completion > junction_completion
                and not stop_infraction
                and not red_light_infraction
            ):
                Ability_Statistic["Traffic_Signs"][0] += 1
                Ability_Statistic["Traffic_Signs"][1] += 1
            else:
                Ability_Statistic["Traffic_Signs"][1] += 1

    Ability_Res = {}
    for ability, statis in Ability_Statistic.items():
        if statis[1] < 1:
            Ability_Res[ability] = 1.0
        else:
            Ability_Res[ability] = float(statis[0]) / float(statis[1])
    Ability_Res["mean"] = sum(list(Ability_Res.values())) / 5

    for key, value in Ability_Res.items():
        print(key, ": ", value)

    return Ability_Res
