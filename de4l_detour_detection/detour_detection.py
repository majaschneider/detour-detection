"""This module detects detours to measure the privacy risk inherent to spatio-temporal trajectories containing stops.
"""
import warnings
from urllib.error import URLError

import pandas as pd
import openrouteservice
from urllib3.exceptions import ConnectTimeoutError
from de4l_geodata.geodata import point as pt
from de4l_geodata.geodata import route as rt
from geopy.exc import GeocoderUnavailable, GeocoderServiceError
from geopy.geocoders import Nominatim


def select_samples_by_temporal_distance(route, temporal_distance, start_timestamp=None):
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
        The temporal distance between sampled points.
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
    for next_idx in range(current_idx + 1, len(route)):
        next_timestamp = route[next_idx].timestamp
        if (next_timestamp - route[current_idx].timestamp) >= temporal_distance:
            sampled_points.append(route[next_idx])
            current_idx = next_idx

    return sampled_points


def select_samples_by_spatial_distance(route, spatial_distance, start_point_idx=0):
    """
    Samples points from a given route with a certain temporal distance. Sampling starts at the first route point or, if
    provided, at the point at index start_point_idx.

    Parameter
    ---------
    route : rt.Route
        A route object containing trajectory data represented as pt.Point. If the parameter route is empty the returned
        route is empty as well. The points in the route are assumed to be sorted according to time. In other words: the
        oldest point is at the first index.
    spatial_distance : float
        The spatial distance in meters between sampled points.
    start_point_idx : int
        The index of the point in route that marks the starting point for selection of samples. Per default, sample
        selection starts at the first route point.

    Returns
    -------
    sampled_points : rt.Route
        A route containing the sampled points according to the spatial distance.
    """
    if not isinstance(route, rt.Route):
        raise ValueError('Wrong value for parameter route. Make sure it is of type de4l_geodata.geodata.route.Route.')

    if not isinstance(spatial_distance, (int, float)) or spatial_distance < 0:
        raise ValueError('Wrong value for parameter spatial_distance. Check that it is a number and greater than zero.')

    sampled_points = rt.Route()
    current_idx = start_point_idx

    # Add first point before iterating
    if current_idx < len(route):
        sampled_points.append(route[current_idx])

    # Iterate over the rest of the route and check whether current and next point are spatial_distance apart
    for next_idx in range(current_idx + 1, len(route)):
        if pt.get_distance(route[current_idx], route[next_idx]) >= spatial_distance:
            sampled_points.append(route[next_idx])
            current_idx = next_idx

    return sampled_points


def sample_from_shape(route, spatial_distance):
    """
    Creates new points along the shape of the indicated route such that a certain spatial distance is kept. Sampling
    starts at the first route point. The sampled points are interpolated along the shape of the input route, which means
    in turn, that they need not necessarily match the road network. If the latter is required, the sampled points need
    to be map-matched afterwards.

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

    # read segment per segment and check its length
    remaining_distance_to_go = spatial_distance
    for index in range(1, len(route)):
        segment_start = route[index - 1]
        segment_end = route[index]
        # if segment_end is the route's last point and its length is smaller than the remaining_distance_to_go
        if segment_end is last_point and remaining_distance_to_go > pt.get_distance(segment_start, segment_end):
            break
        while remaining_distance_to_go <= pt.get_distance(segment_start, segment_end):
            # sample on the current segment
            ratio_interpolated_point = remaining_distance_to_go / pt.get_distance(segment_start, segment_end)
            interpolated_point = pt.get_interpolated_point(segment_start, segment_end, ratio_interpolated_point)
            sampled_points.append(interpolated_point)
            segment_start = interpolated_point
            remaining_distance_to_go = spatial_distance
        # advance the remaining spatial distance on the current segment
        remaining_distance_to_go -= pt.get_distance(segment_start, segment_end)
    return sampled_points


def reverse_geocode(route, nominatim_url, scheme='https'):
    """
    This function iterates over a route consisting of points with longitude and latitude values and converts them into
    real-world addresses using Nominatim's `reverse` function. This process is called 'reverse geocoding'. The `reverse`
    function maps coordinates to the road network by assigning them to the closest street. Nominatim is Latin for
    'by name'. It is an open source tool that can also be used to search Open Street Map (OSM) data by name and to
    generate synthetic addresses out of OSM points. We run a local instance of Nominatim. For instructions on
    deployment, take a look at the README in the repository root.


    Further resources:

    Nominatim: https://nominatim.org/

    Documentation `reverse`: https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.reverse

    Parameters
    ----------
    route : rt.Route
        A route containing points that should be reversed.
    nominatim_url : str
        The url pointing to a running instance of Nominatim.
    scheme : str
        The protocol to be used for openrouteservice queries, e.g. http or https.

    Returns
    -------
    reverse_geocoded_points : rt.Route
        A route that contains of reverse geocoded points represented as pt.Point. The points in the route have the same
        coordinate unit as the input points in the input route e.g. 'degrees'.
    failed_requests : int
        The number of requests to Nominatim that failed either because of a missing connection or because of wrong
        point coordinates.
    """
    nominatim = Nominatim(scheme=scheme, domain=nominatim_url)
    reverse_geocoded_points = rt.Route()
    failed_requests = 0
    if len(route) > 0:
        input_coordinate_unit_is_radians = route[0].get_coordinates_unit() == 'radians'
        for point in route:
            point = point.to_degrees(ignore_warning=True)
            reverse_geocoded_point = nominatim_reverse(point, nominatim)
            # Check whether a request to Nominatim succeeded. A request might fail due to connection issues or because
            # there are no results for a given point
            if isinstance(reverse_geocoded_point, pt.Point):
                if input_coordinate_unit_is_radians:
                    reverse_geocoded_point.to_radians_()
                reverse_geocoded_points.append(reverse_geocoded_point)
            else:
                failed_requests += 1

        if failed_requests > 0:
            warnings.warn(f'Reverse geocoding failed for {failed_requests} of {len(route)} points. Make sure, that the '
                          f'Nominatim service is available and serving the correct map data and that the provided '
                          f'points are in the correct format.')

    return reverse_geocoded_points, failed_requests


def nominatim_reverse(point, nominatim):
    """
    This function calls Nominatim's `reverse` function to convert a point into a real-world address. This is achieved by
    mapping locations to their closest street segment. The function returns None in the case that a connection to
    Nominatim can't be established or the provided point can not be processed by Nominatim.


    Further resources:

    Documentation reverse: https://geopy.readthedocs.io/en/stable/#geopy.geocoders.Nominatim.reverse

    Parameters
    ----------
    point : pt.Point
        The point to be reversed.
    nominatim : Nominatim
        An instance of Nominatim that is ready to accept requests.

    Returns
    -------
    reversed_point : pt.Point or None
        In case reverse geocoding succeeds, a Point is returned. Otherwise, an error is raised. The output point has
        the same coordinates unit as the input.

    """
    reversed_location = None
    input_coordinates_unit_is_radians = point.get_coordinates_unit() == 'radians'
    # Nominatim requires latitude/longitude in 'degrees' format
    point = point.to_degrees(ignore_warnings=True)
    nominatim_input = [point.y_lat, point.x_lon]

    try:
        reversed_location = nominatim.reverse(nominatim_input)
    except (ValueError,
            ConnectTimeoutError,
            GeocoderUnavailable,
            ConnectionRefusedError,
            URLError,
            GeocoderServiceError) as error:
        warnings.warn(f'Failed to reverse with Nominatim, the following error occurred: {error}')

    if reversed_location is None:
        raise ValueError(f'Failed to reverse with Nominatim, the result for point {point} was empty. Please check the '
                         f'coordinates and the map data of Nominatim.')
    reversed_point = pt.Point([reversed_location.longitude, reversed_location.latitude], coordinates_unit='degrees')
    if input_coordinates_unit_is_radians:
        reversed_point.to_radians_()

    return reversed_point


def get_directions_for_points(start, end, openrouteservice_client, openrouteservice_profile='driving-car',
                              average_speed_kmh=45):
    """
    Use Openrouteservice to get directions between start and end. Directions are instructions within the traffic system
    that guide an actor towards a destination. A profile can be used to specify this actor's transport mode.

    Parameters
    ----------
    start : pt.Point
        A geographical point marking the beginning of the route to calculate.
    end : pt.Point
        A geographical point marking the end of the route to calculate.
    openrouteservice_client : openrouteservice.client.Client
        A running instance of the Openrouteservice client.
    openrouteservice_profile :
        {'driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
        'cycling-road', 'cycling-mountain', 'cycling-electric'}
        Specifies the mode of transport to use when calculating directions.
    average_speed_kmh : float
        The default speed in kilometers per hour, that is assumed, in case the duration of a route could not be
        retrieved and needs to be estimated. The default value is representing the average speed of a car in a city.

    Returns
    -------
    directions : dict
        Directions between start and end, containing
        distance : float
            The distance of the shortest route between start and end in meters.
        duration : float
            The duration when moving along the shortest route between start and end with the given transport mode in
            seconds.
        route : rt.Route
            The shortest route between start and end in the same format as the start point.
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

    start_coordinates_unit_is_radians = start.get_coordinates_unit() == 'radians'
    start = start.to_degrees(ignore_warnings=True)
    end = end.to_degrees(ignore_warnings=True)
    route = rt.Route([start, end])
    ors_response = None

    shortest_route = None
    distance = None
    duration = None

    try:
        ors_response = \
            openrouteservice_client.directions((start, end), profile=openrouteservice_profile, format='geojson')
    except Exception:
        # exceptions are ignored and assumed values will be provided instead (direct connection between start and end)
        pass

    if ors_response is not None:
        # ors response can contain multiple nodes (each node contains details for one shortest route proposition)
        all_nodes = ors_response['features']

        # get all nodes that have a distance (which is sometimes missing)
        nodes_with_distance = [node for node in all_nodes if 'distance' in node['properties']['summary'].keys()]

        # if multiple nodes available, choose the one with the shortest distance, if available at all
        distances = [node['properties']['summary']['distance'] for node in nodes_with_distance]
        if len(distances) > 0:
            min_distance = min(distances)
            chosen_node = \
                [node for node in nodes_with_distance if node['properties']['summary']['distance'] == min_distance][0]
        else:
            # if distance not available at all, choose the first available node and calculate the missing parameters
            chosen_node = all_nodes[0]
        shortest_route = rt.Route([
            pt.Point(point, coordinates_unit='degrees') for point in chosen_node['geometry']['coordinates']
        ])

        route_properties = chosen_node['properties']['summary']
        if 'distance' in route_properties.keys():
            distance = route_properties['distance']

        if 'duration' in route_properties.keys():
            duration = route_properties['duration']

    if shortest_route is None:
        warnings.warn(f'Shortest route could not be retrieved for {route}. The input route is returned.')
        shortest_route = route
    if distance is None:
        distance = pt.get_distance(start, end)
        warnings.warn('Distance is not available. The direct connection (as the crow flies) will be used.')
    if duration is None:
        duration = distance / (average_speed_kmh * 1_000 / 3_600)
        warnings.warn(f'Duration is not available and will be estimated based on an assumed speed of '
                      f'{average_speed_kmh} km/h.')

    if start_coordinates_unit_is_radians:
        shortest_route.to_radians_()

    return {'route': shortest_route, 'distance': distance, 'duration': duration}


def get_directions_for_route(route, openrouteservice_base_path, scheme='https', openrouteservice_profile='driving-car'):
    """
    Calculate directions between all consecutive pairs of geographical points of a route. This is done by using
    Openrouteservice and its `directions` functionality. Directions are instructions within the traffic system that
    guide an actor towards a destination. A profile can be used to specify this actor's transport mode.

    Parameters
    ----------
    route : rt.Route
        The route to get directions for.
    openrouteservice_base_path : str
        The base address of an available instance of Openrouteservice. Its structure should be '[host]:[port]'.
    scheme : str
        The protocol to be used for openrouteservice queries, e.g. http or https.
    openrouteservice_profile :
        {'driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
        'cycling-road', 'cycling-mountain', 'cycling-electric'}
        Specifies the mode of transport to use when calculating directions.
        See: https://openrouteservice-py.readthedocs.io/en/latest/#module-openrouteservice.directions

    Returns
    -------
    directions_for_route : List
        A list containing directions connecting every consecutive pair of points from route. Every entry is a dict
        that contains the keys 'distance', 'duration' and 'route' for the shortest route between the respective start
        and end point.
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

    ors_url = f'{scheme}://{openrouteservice_base_path}/ors'
    client = openrouteservice.Client(base_url=ors_url)

    directions_for_route = []
    end_idx = len(route) - 1
    for idx in range(end_idx):
        start = route[idx]
        end = route[idx + 1]
        directions = get_directions_for_points(start,
                                               end,
                                               openrouteservice_client=client,
                                               openrouteservice_profile=openrouteservice_profile)
        directions_for_route.append(directions)

    return directions_for_route
