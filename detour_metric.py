"""This module detects detours to measure the privacy risk inherent to spatio-temporal trajectories containing stops.
"""

from geodata import route as rt
import pandas as pd


def select_samples(route, temporal_distance, start_timestamp=None):
    """
    Samples points from a given route with a certain temporal distance. Sampling starts at the first route point or, if
    provided, at start_timestamp. The route must contain timestamps. The function assumes that the temporal distances
    between points in the route are equal. Dealing with irregular timestamps will be addressed later.

    Parameter
    ---------
    route : geodata.route.Route
        A route object containing trajectory data represented as geodata.point_t.PointT. If the parameter route is empty
        the returned route is empty as well. The points in the route are assumed to be sorted according to timestamp. In
         other words: the oldest timestamp is at the first index.
    temporal_distance : pandas.Timedelta
        The temporal distance in seconds between sampled points.
    start_timestamp : pandas.Timestamp
        An optional timestamp that marks the starting point for selection of samples. If None, sampling starts at the
        first route point.
    Returns
    -------
    sampled_points : geodata.route.Route
        A route containing the sampled points according to the temporal distance.
    """
    if not (isinstance(route, rt.Route) and route.has_timestamps()):
        raise ValueError(
            "Wrong value for parameter route. Make sure it is of type route.Route and its points have timestamps."
         )

    if not isinstance(temporal_distance, pd.Timedelta) or temporal_distance.seconds < 0:
        raise ValueError(
            "Wrong value for parameter temporal distance. Check if it is of type pandas.Timedelta and larger than zero."
        )

    sampled_points = rt.Route()
    current_idx = 0

    # If a start timestamp is given, iterate over the route until it is reached
    if start_timestamp is not None:
        if not isinstance(start_timestamp, pd.Timestamp):
            raise ValueError("Wrong value for parameter start timestamp. Make sure it is of type pandas.Timestamp.")
        for point in route:
            if point.timestamp < start_timestamp:
                current_idx += 1

    # Add first point before iterating.
    if current_idx < len(route):
        sampled_points.append(route[current_idx])

    # Iterate over the rest of the route and check whether current and next point's timestamps are temporal_distance
    # apart
    for next_idx in range(current_idx+1, len(route)):
        next_timestamp = route[next_idx].timestamp
        if (next_timestamp-route[current_idx].timestamp) >= temporal_distance:
            sampled_points.append(route[next_idx])
            current_idx = next_idx

    return sampled_points
