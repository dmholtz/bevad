import re

from bevad_sim.data_interface.core_container import CoreContainer

PENALTY_COLLISION_LAYOUT = 0.65
PENALTY_COLLISION_PEDESTRIAN = 0.5
PENALTY_COLLISION_VEHICLE = 0.6
PENALTY_STOP_INFRACTION = 0.8
PENALTY_SCENARIO_TIMEOUT = 0.7
PENALTY_TRAFFIC_LIGHT_INFRACTION = 0.7
PENALTY_YIELD_TO_EMERGENCY_VEHICLE = 0.7


def get_outside_lane_percent(desc: str) -> float:
    """Parse the textual outside lane infraction description to get the percentage of route completed outside lane."""
    match = re.search(r"\(([\d.]+)%", desc)
    if match:
        return float(match.group(1))
    else:
        return 0.0


def compute_b2d_metrics(episode: CoreContainer) -> dict:
    info = episode.step_meta.info[0][-1]["lb_metrics"]

    route_completion = info["score_route"]

    def get_outside_lane_penalty(desc: str):
        match = re.search(r"\(([\d.]+)%", desc)
        if match:
            return 1 - float(match.group(1)) / 100
        else:
            return 1

    # infractions
    penalty_collision_layout = [
        PENALTY_COLLISION_LAYOUT for _ in info["collisions_layout"]
    ]
    penalty_collision_pedestrian = [
        PENALTY_COLLISION_PEDESTRIAN for _ in info["collisions_pedestrian"]
    ]
    penalty_collision_vehicle = [
        PENALTY_COLLISION_VEHICLE for _ in info["collisions_vehicle"]
    ]
    penalty_stop = [PENALTY_STOP_INFRACTION for _ in info["stop_infraction"]]
    penalty_traffic_light = [
        PENALTY_TRAFFIC_LIGHT_INFRACTION for _ in info["red_light"]
    ]
    penalty_scenario_timeout = [
        PENALTY_SCENARIO_TIMEOUT for _ in info["scenario_timeouts"]
    ]
    penalty_yield_emergency = [
        PENALTY_YIELD_TO_EMERGENCY_VEHICLE
        for _ in info["yield_emergency_vehicle_infractions"]
    ]
    penalty_outside_lane = [
        get_outside_lane_penalty(infraction)
        for infraction in info["outside_route_lanes"]
    ]

    penalties = [
        *penalty_collision_layout,
        *penalty_collision_pedestrian,
        *penalty_collision_vehicle,
        *penalty_stop,
        *penalty_traffic_light,
        *penalty_scenario_timeout,
        *penalty_yield_emergency,
        *penalty_outside_lane,
    ]

    penalty = 1
    for p in penalties:
        penalty *= p

    if penalty >= 1 and route_completion >= 100:
        success = True
    else:
        success = False

    return {
        "success": 1 if success else 0,
        "driving_score": penalty * route_completion,
        "route_completion": route_completion,
        "infraction_score": penalty,
    }


def compute_val13_score(episode: CoreContainer) -> dict:
    info = episode.step_meta.info[0][-1]["lb_metrics"]

    # route completion adjusted for outside lane driving
    route_completion_total = info["score_route"]
    outside_lane = sum(
        [
            get_outside_lane_percent(infraction)
            for infraction in info["outside_route_lanes"]
        ]
    )
    route_completion = route_completion_total - outside_lane

    # regular driving score (see Leaderboard 2.0)
    infraction_score = PENALTY_COLLISION_VEHICLE ** len(info["collisions_vehicle"])
    infraction_score *= PENALTY_COLLISION_PEDESTRIAN ** len(
        info["collisions_pedestrian"]
    )
    infraction_score *= PENALTY_COLLISION_LAYOUT ** len(info["collisions_layout"])
    infraction_score *= PENALTY_STOP_INFRACTION ** len(info["stop_infraction"])
    infraction_score *= PENALTY_TRAFFIC_LIGHT_INFRACTION ** len(info["red_light"])
    infraction_score *= PENALTY_YIELD_TO_EMERGENCY_VEHICLE ** len(
        info["yield_emergency_vehicle_infractions"]
    )
    infraction_score *= PENALTY_SCENARIO_TIMEOUT ** len(info["scenario_timeouts"])
    regular_score = route_completion * infraction_score

    # normalized driving score (see HiddenBiasesDatasets paper)
    route_km = info["route_length"] / 1000
    infraction_coefficient = (0.2 * PENALTY_COLLISION_VEHICLE) ** (
        len(info["collisions_vehicle"]) / route_km
    )
    infraction_coefficient *= (0.2 * PENALTY_COLLISION_PEDESTRIAN) ** (
        len(info["collisions_pedestrian"]) / route_km
    )
    infraction_coefficient *= (0.2 * PENALTY_COLLISION_LAYOUT) ** (
        len(info["collisions_layout"]) / route_km
    )
    infraction_coefficient *= (0.2 * PENALTY_STOP_INFRACTION) ** (
        len(info["stop_infraction"]) / route_km
    )
    infraction_coefficient *= (0.2 * PENALTY_TRAFFIC_LIGHT_INFRACTION) ** (
        len(info["red_light"]) / route_km
    )
    infraction_coefficient *= (0.2 * PENALTY_YIELD_TO_EMERGENCY_VEHICLE) ** (
        len(info["yield_emergency_vehicle_infractions"]) / route_km
    )
    infraction_coefficient *= (0.2 * PENALTY_SCENARIO_TIMEOUT) ** (
        len(info["scenario_timeouts"]) / route_km
    )
    normalized_score = route_completion * infraction_coefficient

    # success rate
    if infraction_coefficient >= 1 and route_completion >= 100:
        success = True
    else:
        success = False

    return {
        "success": 1 if success else 0,
        "driving_score": normalized_score,
        "regular_driving_score": regular_score,
        "route_completion": route_completion,
        "infraction_score": infraction_score,
        "infraction_coefficient": infraction_coefficient,
    }


def compute_longest6_score(episode: CoreContainer) -> dict:
    info = episode.step_meta.info[0][-1]["lb_metrics"]

    # route completion adjusted for outside lane driving
    route_completion_total = info["score_route"]
    outside_lane = sum(
        [
            get_outside_lane_percent(infraction)
            for infraction in info["outside_route_lanes"]
        ]
    )
    route_completion = route_completion_total - outside_lane

    # driving score (see Transfuser paper)
    infraction_score = PENALTY_COLLISION_VEHICLE ** len(info["collisions_vehicle"])
    infraction_score *= PENALTY_COLLISION_PEDESTRIAN ** len(
        info["collisions_pedestrian"]
    )
    infraction_score *= PENALTY_COLLISION_LAYOUT ** len(info["collisions_layout"])
    infraction_score *= PENALTY_TRAFFIC_LIGHT_INFRACTION ** len(info["red_light"])
    driving_score = route_completion * infraction_score

    # success rate
    if infraction_score >= 1 and route_completion >= 100:
        success = True
    else:
        success = False

    return {
        "success": 1 if success else 0,
        "driving_score": driving_score,
        "route_completion": route_completion,
        "infraction_score": infraction_score,
    }
