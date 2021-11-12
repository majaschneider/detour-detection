"""This module detects detours to measure the privacy risk inherent to spatio-temporal trajectories containing stops.
"""
import requests
from math import degrees
from urllib3.exceptions import ConnectTimeoutError

import pandas as pd
import openrouteservice
from geodata import route as rt
from geodata import point as pt
from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
from geopy.location import Location


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
    if not isinstance(route, rt.Route) or not route.has_timestamps():
        raise(ValueError(
            "Wrong value for param route. Make sure it is of type route.Route and its points have timestamps."
         ))

    if not isinstance(temporal_distance, pd.Timedelta) or temporal_distance.seconds < 0:
        raise(ValueError(
            "Wrong value for param temporal distance. Make sure it is of type pandas.Timedelta and larger than zero."
        ))

    sampled_points = rt.Route()
    current_idx = 0

    # If a start timestamp is given, iterate over the route until it is reached
    if start_timestamp is not None:
        if not isinstance(start_timestamp, pd.Timestamp):
            raise ValueError("Wrong value for start timestamp. Make sure it is of type pandas.Timestamp.")
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


def nominatim_reverse(point, nominatim):
    """
    This function calls Nominatim's reverse function to convert a coordinate consisting of longitude and latitude in
    degrees format into a Location. The function returns None in case a connection to Nominatim can't be established or
    a ValueError is raised.
    Further resources:
    Documentation reverse: https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.reverse

    Parameters
    ----------
    point : list
        A geographical point consisting of longitude and latitude in degree format represented as a list.
    nominatim : geopy.geocoders.Nominatim
        An instance of Nominatim that is ready to accept requests.

    Returns
    -------
    reversed_point : geopy.location.Location or None
        In case reverse geocoding succeeds, a location is returned. Otherwise, None is returned.

    """
    reversed_point = None

    try:
        reversed_point = nominatim.reverse(point)
    except (ValueError, ConnectTimeoutError, GeocoderUnavailable) as error:
        if isinstance(error, ValueError):
            print("The latitude and longitude values of the point are not valid.")
        else:
            print("Connection to Nominatim could not be established.")

    return reversed_point


def reverse_geocode(sampled_points, nominatim_url):
    """
    This function maps sampled geographical points to real-world geographical Locations using Nominatim. It establishes
    a connection to a running instance of Nominatim and uses nominatim_reverse to actually convert points. The function
    counts two kinds of error occurrences for analytical purposes:
    - failed_requests is the number of instances that were passed to nominatim_reverse but could not be converted by
      Nominatim
    - wrong_values is the number of elements in route that are not of type geodata.point.Point.
    Further resources:
    Nominatim: https://nominatim.org/
    Documentation reverse: https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.reverse

    Parameters
    ----------
    sampled_points : geodata.route.Route
        A route containing geographical points of type geodata.point.Point.
    nominatim_url : str
        The url of the available Nominatim instance.

    Returns
    -------
    reverse_geocoded_points : list
        A list of reverse geocoded points represented as geopy.location.Location.
    failed_requests : int
        The number of requests against Nominatim that failed either because of a missing connection or because of wrong
        point coordinates.
    wrong_values : int
        The number of points in the route that were are instances of geodata.point.Point.
    """
    nominatim = Nominatim(scheme="http", domain=nominatim_url)
    reverse_geocoded_points = []
    failed_requests = 0
    wrong_values = 0
    for point in sampled_points:
        try:
            # Convert to degrees, as Nominatim works with degrees
            point_deg = [degrees(point.x_lon), degrees(point.y_lat)]
            reverse_geocoded_point = nominatim_reverse(point_deg, nominatim)
            if isinstance(reverse_geocoded_point, Location):
                reverse_geocoded_points.append(reverse_geocoded_point)
            else:
                failed_requests += 1
        except AttributeError:
            wrong_values += 1

    if failed_requests > 0:
        print(f"Failed requesting to reverse geocode {failed_requests} points. Check if you are able to connect to "
              f"Nominatim, that your instance has the correct map data and that your points' values are valid.")
    if wrong_values > 0:
        print(f"Failed to query Nominatim for {wrong_values} points. Check if these are of type geodata.point.Point.")

    return reverse_geocoded_points, failed_requests, wrong_values


def get_optimal_route(start, end, ors_client):
    """
    Use Openrouteservice to get a connecting route between two geographical points.

    Parameters
    ----------
    start : geodata.point.Point
        A geographical point marking the beginning of the route to calculate.
    end : geodata.point.Point
        A geographical point marking the end of the route to calculate.
    ors_client : openrouteservices.client.Client
        A running instance of the Openrouteservice client.

    Returns
    -------
    route : Geojson
        A route between start and end in Geojson format.

    """
    if not isinstance(start, pt.Point) or not isinstance(end, pt.Point):
        raise ValueError(
            "Failed to get a optimal route for a point. Make sure the route contains values of type geodata.point.Point"
            "."
        )

    coords = ((degrees(start[0]), degrees(start[1])), (degrees(end[0]), degrees(end[1])))

    try:
        ors_response = ors_client.directions(coords)
        routes = ors_response["routes"]
        if len(routes) == 1:
            time_optimal_route_package = routes[0]
        else:
            times = [r["summary"]["duration"] for r in routes]
            min_time = min(times)
            time_optimal_route_package = [r for r in routes if r["summary"]["duration"] == min_time][0]
        time_optimal_route = time_optimal_route_package["segments"][0]
    except openrouteservice.exceptions.ApiError:
        time_optimal_route = {}

    return time_optimal_route


def get_optimal_routes(sampled_points, openrouteservice_base_path):
    """
    Calculate the optimal connections between consecutive pairs of geographical points. This is done by using
    Openrouteservice and its directions functionality.

    Parameters
    ----------
    sampled_points : geodata.route.Route
        A route containing geographical points sampled using a given temporal distance.
    openrouteservice_base_path : str
        The base address of the available instance of Openrouteservice. Its structure should be like 'localhost:8001'.

    Returns
    -------
    optimal_route : list
        A list containing the optimal routes connecting every consecutive pair of points from the input in Geojson
        format.

    """
    if not isinstance(sampled_points, rt.Route) or len(sampled_points) == 0:
        raise ValueError(
            "Failed to get optimal routes. Make sure to pass a non-empty route of type geodata.route.Route."
        )

    ors_url = "http://" + openrouteservice_base_path + "/ors"
    client = openrouteservice.Client(base_url=ors_url)

    optimal_routes = []
    for idx in range(1, len(sampled_points)):
        start = sampled_points[idx - 1]
        end = sampled_points[idx]
        route = get_optimal_route(start, end, ors_client=client)
        optimal_routes.append(route)

    return optimal_routes


def calculate_difference_by_duration(sampled_points, openrouteservice_base_path):
    """
    Calculate the difference in duration between the estimated time of directions given by Openrouteservice and the real
    time that passed between two consecutive points.

    Parameters
    ----------
    sampled_points : geodata.route.Route
        A route consisting of points with timestamps, which a have been sampled given a certain temporal distance as
        sampling rate.
    openrouteservice_base_path : str
        The base address of the available instance of Openrouteservice. Its structure should be like 'localhost:8001'.

    Returns
    -------
    differences : list
        A list where each entry is a float representing the difference between actual time between two consecutive
        points and time as of a route between these two points.

    """
    differences = []

    routes = get_optimal_routes(sampled_points, openrouteservice_base_path)

    for idx in range(1, len(sampled_points)):
        start_point = sampled_points[idx-1]
        end_point = sampled_points[idx]
        route = routes[idx-1]

        # get timestamp difference between start and end
        duration_points = (end_point.timestamp - start_point.timestamp).seconds
        # get duration of optimal route
        duration_route = route["duration"]
        # calculate the difference and add to result
        differences.append(duration_points - duration_route)

    return differences


def get_samples_between(sampled_points, start, end, temporal_distance, k):
    """
    Get samples between start and end and avoid last match if it would sting end.
    Idea: trim the list of all timestamps that are less or equal than end-temporal_distance earlier than end timestamp.
    Add the last point again.
    Furthermore check whether enough samples are retrieved:

    Parameters
    ----------
    sampled_points : geodata.route.Route
        A route consisting of points with timestamps, which a have been sampled given a certain temporal distance as
        sampling rate.
    start : geodata.point.PointT
        The point that should be the first point in the returned route.
    end : geodata.point.PointT
        The point that should be the last point in the returned route.
    temporal_distance : pandas.timedelta.Timedelta
        The initial time difference between two consecutive points that is given in the parameter sampled_points. It is
         used to derive the smaller temporal difference.
    k : int
        The divisor that is used to determine the second, smaller temporal distance for sub sampling. It furthermore
        indicates the maximum number of samples that can be retrieved during sub sampling, k + 1.

    Returns
    -------
    samples_between : geodata.route.Route
        A route consisting of points between starting and ending which are at least temporal distance apart. Starting
        and ending point are included.

    """
    temporal_distance = temporal_distance / k
    max_new_samples = k + 1

    samples_reduced = rt.Route([p for p in sampled_points if p.timestamp < (end.timestamp - temporal_distance)])
    samples_between = select_samples(samples_reduced, temporal_distance, start_timestamp=start.timestamp)
    samples_between.append(end)

    if len(samples_between) < (max_new_samples / 2):
        return rt.Route([])
    else:
        return samples_between


def calculate_difference_by_distance_and_duration(sampled_points, openrouteservice_base_path, temporal_distance, k):
    """
    This function calculates the differences in duration and distance between pairs of consecutive geographical points.
    Therefore, the function samples points a second time, with a second, smaller temporal distance. This is calculated
    using k.

    Parameters
    ----------
    sampled_points : geodata.route.Route
        A route consisting of points with timestamps, which a have been sampled given a certain temporal distance as
        sampling rate.
    openrouteservice_base_path : str
        The base address of the available instance of Openrouteservice. Its structure should be like 'localhost:8001'.
    temporal_distance : pandas.Timedelta
        The temporal distance in seconds between sampled points.
    k : int
        The divisor that is used to determine the second, smaller temporal distance for sub sampling.

    Returns
    -------
    differences : list
        A list of dicts containing the difference in distance and in duration between optimal routes connecting two
        consecutive points and routes connecting a set of sub sampled points between these two points.

    """
    if not isinstance(k, int) or k < 3:
        raise ValueError("Wrong value for parameter k for sub sampling value. Make sure it is an integer larger than 2.")

    differences = []

    samples_initial_rate = select_samples(sampled_points, temporal_distance)

    optimal_routes_initial_rate = get_optimal_routes(samples_initial_rate, openrouteservice_base_path)

    for idx in range(1, len(samples_initial_rate)):
        start_point = samples_initial_rate[idx-1]
        end_point = samples_initial_rate[idx]
        optimal_route_distance = optimal_routes_initial_rate[idx-1]["distance"]
        optimal_route_duration = optimal_routes_initial_rate[idx-1]["duration"]

        samples_smaller_rate = get_samples_between(sampled_points=sampled_points,
                                                   start=start_point,
                                                   end=end_point,
                                                   temporal_distance=temporal_distance,
                                                   k=k)
        optimal_routes_smaller_rate = get_optimal_routes(samples_smaller_rate, openrouteservice_base_path)
        summed_distance = sum(directions["distance"] for directions in optimal_routes_smaller_rate)
        summed_duration = sum(directions["duration"] for directions in optimal_routes_smaller_rate)

        delta_distance = summed_distance - optimal_route_distance
        delta_duration = summed_duration - optimal_route_duration

        differences.append({"distance_difference": delta_distance, "duration_difference": delta_duration})

    return differences
