from agents.navigation.local_planner import RoadOption


def downsample_route(route, sample_factor):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # At the beginning
        if (
            prev_option is None
            or curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT)
            or (
                prev_option != curr_option
                and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT)
                or dist > sample_factor
            )
            or i == len(route) - 1
        ):
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i - 1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample
