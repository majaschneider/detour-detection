"""This module detects detours to measure the privacy risk inherent to spatio-temporal trajectories containing stops.
"""
from math import degrees
from urllib.error import URLError

import pandas as pd
import openrouteservice
from urllib3.exceptions import ConnectTimeoutError
from de4l_geodata.geodata import point as pt
from de4l_geodata.geodata import route as rt
from geopy.exc import GeocoderUnavailable, GeocoderServiceError
from geopy.geocoders import Nominatim
from geopy.location import Location


def select_samples(route, temporal_distance, start_timestamp=None):
    """
    Selects points of a given route so that a certain temporal distance is kept. Sampling starts at the first route
    point or, if provided, at start_timestamp. The route points must contain timestamps. The function assumes that the
    temporal distances between points in the route are equal. Dealing with irregular timestamps will be addressed later.

    Parameter
    ---------
    route : rt.Route
        A route object containing trajectory data represented as de4l_geodata.geodata.point_t.PointT. If the parameter
        route is empty the returned route is empty as well. The points in the route are assumed to be sorted ascending
        by their timestamp.
    temporal_distance : pd.Timedelta
        The temporal distance in seconds between sampled points.
    start_timestamp : pd.Timestamp
        An optional timestamp that marks the starting point for selection of samples. If None, sampling starts at the
        first route point.

    Returns
    -------
    sampled_points : rt.Route
        A route containing the sampled points according to the temporal distance.
    """
    if not (isinstance(route, rt.Route) and route.has_timestamps()):
        raise ValueError(
            'Wrong value for parameter route. Make sure it is of type de4l_geodata.geodata.route.Route and its points '
            'have timestamps.'
         )

    if not isinstance(temporal_distance, pd.Timedelta) or temporal_distance.seconds < 0:
        raise ValueError(
            'Wrong value for parameter temporal distance. Check if it is of type pandas.Timedelta and larger than zero.'
        )

    sampled_points = rt.Route()
    current_idx = 0

    # If a start timestamp is given, iterate over the route until it is reached
    if start_timestamp is not None:
        if not isinstance(start_timestamp, pd.Timestamp):
            raise ValueError('Wrong value for parameter start timestamp. Make sure it is of type pandas.Timestamp.')
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


def sample_from_shape(route, spatial_distance):
    """
    Creates new points along the shape of the indicated route such that a certain spatial distance is kept. Sampling
    starts at the first route point. The route points must contain timestamps. The sampled points are interpolated
    along the shape of the input route, which means in turn, that they need not necessarily match the road network. If
    the latter is required, the sampled points need to be map-matched afterwards.

    Parameter
    ---------
    route : rt.Route
        A route containing geographical points in radians and 'latlon' format, indicating a trajectory. If the route
        is shorter than the spatial_distance or has less than two points, an empty route is returned.
    spatial_distance : float
        The spatial distance in meters between each pair of consecutive sampled points.

    Returns
    -------
    sampled_points : rt.Route
        A route containing geographical points in radians and 'latlon' format, sampled from input route at a rate of
        spatial_distance.
    """
    if not (isinstance(route, rt.Route) and len(route) > 1):
        raise ValueError("Wrong value for parameter route. Make sure it is of type route.Route and has at least two "
                         "points.")

    # sample first route point
    first_point = route[0].deep_copy()
    last_point = route[len(route) - 1]
    sampled_points = rt.Route([first_point])
    previous_point = first_point

    remaining_spatial_distance = spatial_distance
    for current_point in route:
        # if next route point is the last and the current distance to it is smaller than remaining_spatial_distance
        if current_point is last_point and remaining_spatial_distance > pt.get_distance(previous_point, current_point):
            break
        while remaining_spatial_distance <= pt.get_distance(previous_point, current_point):
            # sample an interpolated point
            ratio_interpolated_point = spatial_distance / pt.get_distance(previous_point, current_point)
            interpolated_point = pt.get_interpolated_point(previous_point, current_point, ratio_interpolated_point)
            sampled_points.append(interpolated_point)
            previous_point = interpolated_point
            remaining_spatial_distance = spatial_distance
        # advance the remaining spatial distance on the current segment
        remaining_spatial_distance -= pt.get_distance(previous_point, current_point)
    return sampled_points


def reverse_geocode(sampled_points, nominatim_url):
    """
    This function iterates over a route consisting of points with latitude and longitude values and converts them into
    real-world addresses using Nominatim's reverse function. This process is called 'reverse geocoding'. The reverse
    function maps coordinates to the road network by assigning them to the closest street. Nominatim is Latin for
    'by name'. It is an open source tool that can also be used to search Open Street Map (OSM) data by name and to
    generate synthetic addresses out of OSM points. We run a local instance of Nominatim. For instructions on
    deployment, take a look at the README in the repository root.
    This function counts and displays two kinds of error occurrences for analytical purposes:
    - failed_requests: The number of instances that were passed to nominatim_reverse but could not be converted by
      Nominatim
    - wrong_values: The number of elements in route that are not of type pt.Point.
    Further resources:
    Nominatim: https://nominatim.org/
    Documentation reverse: https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.reverse

    Parameters
    ----------
    sampled_points : rt.Route
        A route containing geographical points of type pt.Point.
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
        The number of points in the route that were are instances of pt.Point.
    """
    nominatim = Nominatim(scheme='http', domain=nominatim_url)
    reverse_geocoded_points = []
    failed_requests = 0
    wrong_values = 0
    for point in sampled_points:
        # Check whether a point in the route has the correct data type
        if isinstance(point, pt.Point):
            point_deg = [degrees(point.y_lat), degrees(point.x_lon)]
            reverse_geocoded_point = nominatim_reverse(point_deg, nominatim)
            # Check whether a request to Nominatim succeeded. A request might fail due to connection issues or because
            # there are no results for a given point
            if isinstance(reverse_geocoded_point, Location):
                reverse_geocoded_points.append(reverse_geocoded_point)
            else:
                failed_requests += 1
        else:
            wrong_values += 1

    if failed_requests > 0:
        print(f'Failed requesting to reverse geocode {failed_requests} points. Check if you are able to connect to '
              f'Nominatim, that your instance has the correct map data and that the values of your points are valid.')
    if wrong_values > 0:
        print(f'Failed to query Nominatim for {wrong_values} point(s). Check if all points are of type '
              f'de4l_geodata.geodata.point.Point.')

    return reverse_geocoded_points, failed_requests, wrong_values


def nominatim_reverse(point, nominatim):
    """
    This function calls Nominatim's reverse function to convert a coordinate consisting of latitude and longitude in
    degrees format into a real-world address. This is achieved by mapping a coordinate to the closest street.
    The function returns None in case a connection to Nominatim can't be established or a ValueError is raised.
    Further resources:
    Documentation reverse: https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.reverse

    Parameters
    ----------
    point : list
        A geographical point consisting of latitude and longitude in degree format represented as a list.
    nominatim : Nominatim
        An instance of Nominatim that is ready to accept requests.

    Returns
    -------
    reversed_point : Location or None
        In case reverse geocoding succeeds, a location is returned. Otherwise, None is returned.

    """
    reversed_point = None

    try:
        reversed_point = nominatim.reverse(point)
    except (ValueError,
            ConnectTimeoutError,
            GeocoderUnavailable,
            ConnectionRefusedError,
            URLError,
            GeocoderServiceError) as error:
        if isinstance(error, ValueError):
            print('The latitude and longitude values of the point are not valid.')
        else:
            print('Connection to Nominatim could not be established.')

    return reversed_point


def get_directions_for_points(start, end, openrouteservice_client, openrouteservice_profile='driving_car'):
    """
    Use Openrouteservice to get directions between two geographical points. Directions are instructions within the
    traffic system that guide an actor towards a destination. A profile can be used to specify this actor's means of
    moving. Navigation systems are often used to get directions.

    Parameters
    ----------
    start : pt.Point
        A geographical point in radians format marking the beginning of the route to calculate.
    end : pt.Point
        A geographical point in radians format marking the end of the route to calculate.
    openrouteservice_client : openrouteservices.client.Client
        A running instance of the Openrouteservice client.
    openrouteservice_profile :
        {'driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
        'cycling-road', 'cycling-mountain', 'cycling-electric'}
        Specifies the mode of transport to use when calculating directions.


    Returns
    -------
    directions : dict
        Directions between start and end as a dict. The dict contains distance, duration, navigation instructions and a
        linestring of the proposed route between start and end.


    """
    valid_openrouteservice_profiles = ['driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
                                       'cycling-road', 'cycling-mountain', 'cycling-electric']

    if not isinstance(start, pt.Point) or not isinstance(end, pt.Point):
        raise ValueError(
            'Failed to get a route for two points. Make sure the argument route contains values of type '
            'del4_geodata.geodata.point.Point.'
        )
    if not isinstance(openrouteservice_client, openrouteservice.client.Client):
        raise ValueError(
            'Failed to use the Openrouteservice client. Make sure a client for Openrouteservice was passed as argument.'
        )
    if openrouteservice_profile not in valid_openrouteservice_profiles:
        raise ValueError(
            f'The Openrouteservice profile you provided is not valid. Please choose a profile out of '
            f'{valid_openrouteservice_profiles}.'
        )

    coords = ((degrees(start.x_lon), degrees(start.y_lat)), (degrees(end.x_lon), degrees(end.y_lat)))

    try:
        openrouteservice_response = openrouteservice_client.directions(coords,
                                                                       profile=openrouteservice_profile,
                                                                       format='geojson')
        routes = openrouteservice_response['features']
        if len(routes) == 1:
            directions_node = routes[0]
        # if there are multiple routing results available, choose the one with the shortest duration
        else:
            times = [r['properties']['summary']['duration'] for r in routes]
            min_time = min(times)
            directions_node = [r for r in routes if r['properties']['summary']['duration'] == min_time][0]
        # extract the actual directions out of the result
        directions = directions_node['properties']['segments'][0]
        directions['line_string'] = directions_node['geometry']['coordinates']
    except openrouteservice.exceptions.ApiError:
        directions = {}

    return directions


def get_directions_for_route(route, openrouteservice_base_path, openrouteservice_profile='driving-car'):
    """
    Calculate directions between all pairs of geographical points of a route. This is done by using Openrouteservice and
    its directions functionality. Directions are instructions within the traffic system that guide an actor towards a
    destination. A profile can be used to specify this actor's means of moving. Navigation systems are often used to
    get directions.

    Parameters
    ----------
    route : rt.Route
        A route containing geographical points.
    openrouteservice_base_path : str
        The base address of the available instance of Openrouteservice. Its structure should be '[host]:[port]'.
    openrouteservice_profile :
        {'driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
        'cycling-road', 'cycling-mountain', 'cycling-electric'}
        Specifies the mode of transport to use when calculating directions.
        See: https://openrouteservice-py.readthedocs.io/en/latest/#module-openrouteservice.directions

    Returns
    -------
    directions_for_route : list
        A list containing directions connecting every consecutive pair of points from the input. Every entry is a dict
        that contains distance, duration, navigation instructions and a linestring of the proposed route between start
        and end.


    """
    valid_openrouteservice_profiles = ['driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
                                       'cycling-road', 'cycling-mountain', 'cycling-electric']

    if not isinstance(route, rt.Route) or len(route) < 2:
        raise ValueError(
            'Failed to get connecting routes. Make sure to pass a route of type de4l_geodata.geodata.route.Route that '
            'contains at least two points.'
        )
    if openrouteservice_profile not in valid_openrouteservice_profiles:
        raise ValueError(
            f'The openrouteservice profile you provided is not valid. Please choose a profile out of '
            f'{valid_openrouteservice_profiles}.'
        )

    ors_url = 'http://' + openrouteservice_base_path + '/ors'
    client = openrouteservice.Client(base_url=ors_url)

    directions_for_route = []
    end_idx = len(route) - 1
    for idx in range(end_idx):
        start = route[idx]
        end = route[idx+1]
        directions = get_directions_for_points(start,
                                               end,
                                               openrouteservice_client=client,
                                               openrouteservice_profile=openrouteservice_profile)
        directions_for_route.append(directions)

    return directions_for_route
