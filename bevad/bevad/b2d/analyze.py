from __future__ import annotations

from bevad_sim.data_interface.core_container import CoreContainer


def build_db():
    import sqlite3

    con = sqlite3.connect("episodes.sqlite")
    cur = con.cursor()
    cur.execute("""
        CREATE table episode(
            id TEXT PRIMARY_KEY,
            town TEXT,
            weather TEXT,
            scenario TEXT,
            duration,
            route_completion,
            score,
            success_rate,
            num_infractions,
            num_collisions_layout,
            num_collisions_pedestrian,
            num_collisions_vehicle,
            num_red_lights,
            num_stop,
            num_outside_lane,
            num_emergency_yield,
            num_scenario_timeouts,
            num_route_dev,
            num_blocked,
            num_route_timeout,
        ) """)
    con.commit()
    return con


def analyze_episode(con, episode: CoreContainer):
    from bevad.b2d.score import compute_b2d_metrics

    id = str(episode.episode_meta.episode_id[0])
    town = str(episode.episode_meta.region[0])
    weather = str(episode.episode_meta.weather[0])
    scenario = episode.episode_meta.scenario_type[0]
    if len(scenario) == 0:
        scenario = "none"
    elif len(scenario) == 1:
        scenario = scenario[0]
    else:
        scenario = str(sorted(scenario))
    duration = float(
        episode.step_meta.timestamps[0, -1] - episode.step_meta.timestamps[0, 0]
    )

    info = episode.step_meta.info[0][-1]["lb_metrics"]
    route_completion = info["score_route"]
    num_infractions = info["num_infractions"]
    num_collisions_layout = len(info["collisions_layout"])
    num_collisions_pedestrian = len(info["collisions_pedestrian"])
    num_collisions_vehicle = len(info["collisions_vehicle"])
    num_red_lights = len(info["red_light"])
    num_stop = len(info["stop_infraction"])
    num_outside_lane = len(info["outside_route_lanes"])
    num_emergency_yield = len(info["yield_emergency_vehicle_infractions"])
    num_scenario_timeouts = len(info["scenario_timeouts"])
    num_route_dev = len(info["route_dev"])
    num_blocked = len(info["vehicle_blocked"])
    num_route_timeout = len(info["route_timeout"])

    metrics = compute_b2d_metrics(episode)

    cur = con.cursor()
    cur.execute(
        "INSERT INTO episode (id, town, weather, scenario, duration, route_completion, score, success_rate, num_infractions, num_collisions_layout, num_collisions_pedestrian, num_collisions_vehicle, num_red_lights, num_stop, num_outside_lane, num_emergency_yield, num_scenario_timeouts, num_route_dev, num_blocked, num_route_timeout) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            id,
            town,
            weather,
            scenario,
            duration,
            route_completion,
            metrics["driving_score"],
            metrics["success"],
            num_infractions,
            num_collisions_layout,
            num_collisions_pedestrian,
            num_collisions_vehicle,
            num_red_lights,
            num_stop,
            num_outside_lane,
            num_emergency_yield,
            num_scenario_timeouts,
            num_route_dev,
            num_blocked,
            num_route_timeout,
        ),
    )
    con.commit()
