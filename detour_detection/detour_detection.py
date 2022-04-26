"""This module detects detours to measure the privacy risk inherent to spatio-temporal trajectories containing stops.
"""
import warnings
from urllib.error import URLError
from decimal import Decimal

import pandas as pd
import numpy as np
import openrouteservice
from urllib3.exceptions import ConnectTimeoutError
from geodata.geodata import point as pt
from geodata.geodata import route as rt
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
        A route object containing trajectory data represented as geodata.geodata.point_t.PointT. If the parameter
        route is empty the returned route is empty as well. The points in the route are assumed to be sorted ascending
        by their timestamp.
    temporal_distance : pd.Timedelta
        The temporal distance between sampled points.
    start_timestamp : pd.Timestamp
        An optional timestamp that marks the starting point for selection of samples. If None, sampling starts at the
        first route point.

    Returns
    -------
    sampled_points_ids : List
        The list of the indices of the points from route, that were sampled according to the temporal distance.
    sampled_points : rt.Route
        A route containing the sampled points according to the temporal distance.
    """
    if not (isinstance(route, rt.Route) and route.has_timestamps()):
        raise ValueError(
            'Wrong value for parameter route. Make sure it is of type geodata.geodata.route.Route and its points '
            'have timestamps.'
        )

    if not isinstance(temporal_distance, pd.Timedelta) or temporal_distance.seconds < 0:
        raise ValueError(
            'Wrong value for parameter temporal distance. Check if it is of type pandas.Timedelta and larger than zero.'
        )

    sampled_points_ids = []
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
        sampled_points_ids.append(current_idx)

    # Iterate over the rest of the route and check whether current and next point's timestamps are temporal_distance
    # apart
    for next_idx in range(current_idx + 1, len(route)):
        next_timestamp = route[next_idx].timestamp
        if (next_timestamp - route[current_idx].timestamp) >= temporal_distance:
            sampled_points_ids.append(next_idx)
            current_idx = next_idx
    sampled_points = rt.Route([route[i] for i in sampled_points_ids])

    return sampled_points_ids, sampled_points


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
    sampled_points_ids : List
        The list of the indices of the points from route, that were sampled according to the spatial distance.
    sampled_points : rt.Route
        A route containing the sampled points according to the spatial distance.
    """
    if not isinstance(route, rt.Route):
        raise ValueError('Wrong value for parameter route. Make sure it is of type geodata.geodata.route.Route.')

    if not isinstance(spatial_distance, (int, float)) or spatial_distance < 0:
        raise ValueError('Wrong value for parameter spatial_distance. Check that it is a number and greater than zero.')

    sampled_points_ids = []
    current_idx = start_point_idx

    # Add first point before iterating
    if current_idx < len(route):
        sampled_points_ids.append(current_idx)

    # Iterate over the rest of the route and check whether current and next point are spatial_distance apart
    for next_idx in range(current_idx + 1, len(route)):
        if pt.get_distance(route[current_idx], route[next_idx]) >= spatial_distance:
            sampled_points_ids.append(next_idx)
            current_idx = next_idx
    sampled_points = rt.Route([route[i] for i in sampled_points_ids])

    return sampled_points_ids, sampled_points


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
    first_point = pt.Point(route[0], coordinates_unit=route[0].get_coordinates_unit())
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


def nominatim_reverse(point, nominatim, zoom_level=17):
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
    zoom_level : {3, 5, 8, 10, 14, 16, 17, 18}
        The level of detail required for the address reversing. Values correspond to following details levels:
            zoom 	address detail
            3 	country
            5 	state
            8 	county
            10 	city
            14 	suburb
            16 	major streets
            17 	major and minor streets
            18 	building

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
    reversed_point = point.deep_copy()
    nominatim_input = [point.y_lat, point.x_lon]

    try:
        reversed_location = nominatim.reverse(nominatim_input, zoom=zoom_level)
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
    reversed_point.set_x_lon(reversed_location.longitude)
    reversed_point.set_y_lat(reversed_location.latitude)
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
            'Failed to get connecting routes. Make sure to pass a route of type geodata.geodata.route.Route that '
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


def calculate_poi_risk(route, spatial_distance, ors_path, ors_scheme, ors_profile, nominatim, sampling_distance=25,
                       acceptable_distance_to_shortest_route=50, exceeding_for_maximum_risk=50, map_matching=True):
    """
    Extracts places of interest from the route by analyzing each route point's risk of belonging to a detour and
    therefore of being a place of interest. The algorithm selects regularly spaced points from the route, so that a
    minimum spatial distance between each sampled point is kept. In the following, the shortest route that connects
    each of the sampled points is calculated via openrouteservice. This route is an assumed optimal route that is
    compared to the original route in order to detect deviations. Points from the original route, that deviate
    by more than an acceptable distance are POI candidates. If openrouteservice is not available, the direct
    connection between points is used as the shortest path. The point of each candidate region of successive connected
    candidates, which deviates the most, is considered a POI.

    Parameters
    ----------
    route : rt.Route
        The route to screen for places of interest.
    spatial_distance : float
        The distance between the points that are selected from the route in order to calculate the shortest route to
        compare the original route with.
    ors_path : str
        The base address of a running instance of Openrouteservice. Its structure should be '[host]:[port]'.
    ors_scheme : str
        The protocol to be used for openrouteservice queries, e.g. http or https.
    ors_profile :
        {'driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
        'cycling-road', 'cycling-mountain', 'cycling-electric'}
        Specifies the mode of transport to use when calculating directions.
        See: https://openrouteservice-py.readthedocs.io/en/latest/#module-openrouteservice.directions
    nominatim : Nominatim
        An instance of Nominatim that is ready to accept requests.
    sampling_distance : float
        The distance in meters between sampled points on the shortest route. The shorter the distance, the more accurate
        the distance calculation between route points and shortest route will be. The longer the distance, the higher
        the computational cost. A recommended value is 25 meters.
    acceptable_distance_to_shortest_route : float
        Distance in meters that a route point is allowed to be apart from the shortest route between selected points,
        without indicating that the point has a high risk of belonging to a detour and therefore being a place of
        interest. If a route point has a deviation from the shortest route that is greater than this threshold, its
        risk will be 100%. A recommended value is 50 meters.
    exceeding_for_maximum_risk : int
        This parameter serves for identifying the risk of a point, by measuring, by how much the point's distance to the
        shortest route exceeds the acceptable_distance_to_shortest_route value. If the exceeding is at least as high as
        exceeding_for_maximum_risk, measured in meters, the risk will be maximal.
    map_matching : bool
        If True, route points will be mapped to a road before further analysis.

    Returns
    -------
    shortest_route : Route
        The shortest route between each consecutive pair of selected route points.
    sampled_routes : Route
        The sampled points from the shortest route.
    selected_points : Route
        The points that were selected to calculate the shortest routes.
    distances_to_shortest_route : List
        A list of the distances of each point to the shortest route.
    risks : List
        A list of the probabilities of each route point to be part of a detour and therefore being a place of interest.
        The greater the distance between route points and the shortest route, the higher the risk. The risk is
        calculated as a ratio of the distance between route point and shortest route and the risk distance threshold.
        If the distance is higher than the threshold, the probability is cut off at 100%.
    pois : rt.Route
        A list of route points, identified as places of interest.
    poi_ids : List
        The indices of the found POIs in the route.
    """
    input_route_coordinates_unit_is_radians = route.get_coordinates_unit() == 'radians'
    if not input_route_coordinates_unit_is_radians:
        route = route.to_radians()

    # match each route point to its closest street segment
    if map_matching:
        route = rt.Route([nominatim_reverse(point, nominatim) for point in route])

    # select points from the route so that the given spatial distance is kept
    selected_points_ids, selected_points = select_samples_by_spatial_distance(route, spatial_distance)
    # add last route point of real route in case it was not selected
    if selected_points[-1] != route[-1]:
        selected_points.append(route[-1])
        selected_points_ids.append(len(route) - 1)

    distances_to_shortest_route = []
    distance_exceedings = []
    risks = []
    shortest_routes = []
    sampled_routes = []
    # for each segment in the selection, calculate the shortest route and compare it to each segment point
    nr_selected_points = len(selected_points)
    for i in range(nr_selected_points - 1):
        segment_end_points = rt.Route([selected_points[i], selected_points[i + 1]])
        # calculate the shortest route between start and end of segment
        shortest_route = get_directions_for_route(segment_end_points, ors_path, ors_scheme, ors_profile)[0]['route']
        shortest_routes.append(shortest_route)
        # sample new points from the shortest route in regular distance
        sampled_route = sample_from_shape(shortest_route, spatial_distance=sampling_distance)
        # append last point from shortest route as it is usually not sampled
        sampled_route.append(shortest_route[-1])
        sampled_routes.append(sampled_route)
        # for each route point in the segment, calculate the distance to the closest sample from the shortest route
        segment_start_idx = selected_points_ids[i]
        segment_end_idx = selected_points_ids[i + 1]
        # ignore first segment point (except for first segment), to avoid comparing end and start points of consecutive
        # segments twice
        first_segment_element = 1 if i > 0 else 0
        segment = rt.Route([route[j] for j in range(segment_start_idx + first_segment_element, segment_end_idx + 1)])
        for point in segment:
            distance_to_shortest_route = \
                min([pt.get_distance(point, sampled_point) for sampled_point in sampled_route])
            distances_to_shortest_route.append(round(distance_to_shortest_route))

            # calculate by how much each point exceeds the acceptable distance to the shortest route
            exceeding = round(distance_to_shortest_route - acceptable_distance_to_shortest_route)
            if exceeding < 0:
                exceeding = 0
            distance_exceedings.append(exceeding)

            # calculate a risk value in [0,1] for realising a color coding in plots
            risk = float(round(Decimal(exceeding / exceeding_for_maximum_risk), 2))
            if risk > 1:
                risk = 1
            risks.append(risk)

    # calculate the POIs based on the distance exceedings
    poi_ids = detect_pois_from_exceedings(distance_exceedings)
    pois = rt.Route([route[idx] for idx in poi_ids])

    if not input_route_coordinates_unit_is_radians:
        for i in range(len(sampled_routes)):
            shortest_routes[i].to_degrees_()
            sampled_routes[i].to_degrees_()
        selected_points.to_degrees_()
        pois.to_degrees_()

    shortest_route = [point for segment in shortest_routes for point in segment]
    sampled_route = [point for segment in sampled_routes for point in segment]

    return shortest_route, sampled_route, selected_points, distances_to_shortest_route, risks, pois, poi_ids


def detect_pois_from_exceedings(distance_exceedings):
    """
    Detect places of interest (POI) based on the exceeding distance over a set distance of points in a route. A point is
    a POI, if it has the maximum exceeding distance in a candidate region of connected points, that all exceed the set
    distance.

    Parameters
    ----------
    distance_exceedings : List
        A list containing the exceeding distance over a maximum distance for each point in route.

    Returns
    -------
    poi_ids : List
        A list of the indices of the distance_exceedings, that have been detected as place of interest.
    """
    poi_ids = []
    in_poi_candidate_region = distance_exceedings[0] > 0
    candidate_distances = []
    candidate_ids = []
    nr_points = len(distance_exceedings)
    for idx in range(nr_points):
        exceeding = distance_exceedings[idx]
        if exceeding > 0:
            candidate_distances.append(exceeding)
            candidate_ids.append(idx)
            in_poi_candidate_region = True
        if in_poi_candidate_region and (exceeding == 0 or idx == nr_points - 1):
            # candidate region has ended, find maximum exceeding and assign first point with max distance as POI
            max_exceeding = np.max(candidate_distances)
            nr_candidates = len(candidate_distances)
            poi = [candidate_ids[i] for i in range(nr_candidates) if candidate_distances[i] == max_exceeding][0]
            poi_ids.append(poi)
            candidate_distances = []
            candidate_ids = []
            in_poi_candidate_region = False
    return poi_ids


def calculate_common_poi_risk(parameters, min_count, route, ors_path, ors_scheme, ors_profile, nominatim,
                              exceeding_for_maximum_risk=50, map_matching=True):
    """
    Calculate POIs with detour-detection approach in multiple parameter settings and select only such POIs that have
    been detected a minimum number of times.

    Parameters
    ----------
    parameters : List
        The parameters of the detour-detection:
        spatial_distance : float
            The distance between the points that are selected from the route in order to calculate the shortest route to
            compare the original route with.
        sampling_distance : float
            The distance in meters between sampled points on the shortest route. The shorter the distance, the more
            accurate the distance calculation between route points and shortest route will be. The longer the distance,
            the higher the computational cost. A recommended value is 25 meters.
        acceptable_distance_to_shortest_route : float
            Distance in meters that a route point is allowed to be apart from the shortest route between selected
            points, without indicating that the point has a high risk of belonging to a detour and therefore being a
            place of interest. If a route point has a deviation from the shortest route that is greater than this
            threshold, its risk will be 100%. A recommended value is 50 meters.
    min_count : int
        The minimum number of times, each POI has to be detected with a parameter setting, before it is rated a POI.
    route : rt.Route
        The route to screen for places of interest.
    ors_path : str
        The base address of a running instance of Openrouteservice. Its structure should be '[host]:[port]'.
    ors_scheme : str
        The protocol to be used for openrouteservice queries, e.g. http or https.
    ors_profile :
        {'driving-car', 'driving-hgv', 'foot-walking', 'foot-hiking', 'cycling-regular',
        'cycling-road', 'cycling-mountain', 'cycling-electric'}
        Specifies the mode of transport to use when calculating directions.
        See: https://openrouteservice-py.readthedocs.io/en/latest/#module-openrouteservice.directions
    nominatim : Nominatim
        An instance of Nominatim that is ready to accept requests.
    exceeding_for_maximum_risk : int
        This parameter serves for identifying the risk of a point, by measuring, by how much the point's distance to the
        shortest route exceeds the acceptable_distance_to_shortest_route value. If the exceeding is at least as high as
        exceeding_for_maximum_risk, measured in meters, the risk will be maximal.
    map_matching : bool
        If True, route points will be mapped to a road before further analysis.

    Returns
    -------
    pois : List
        A list of POIs that have been identified in the route.
    pois_ids_min_count : List
        A list of the indices of the POIs that have been identified in the route.
    """
    all_poi_ids = []
    for (spatial_distance, sampling_distance, acceptable_distance) in parameters:
        shortest_routes, sampled_routes, selected_points, distances_to_shortest_route, _, detected_pois, poi_ids = \
            calculate_poi_risk(route=route,
                               map_matching=map_matching,
                               spatial_distance=spatial_distance,
                               ors_path=ors_path,
                               ors_scheme=ors_scheme,
                               ors_profile=ors_profile,
                               nominatim=nominatim,
                               sampling_distance=sampling_distance,
                               acceptable_distance_to_shortest_route=acceptable_distance,
                               exceeding_for_maximum_risk=exceeding_for_maximum_risk)
        all_poi_ids.append(poi_ids)

    # select only pois that have been detected multiple times
    poi_ids_flat = [poi for poi_list in all_poi_ids for poi in poi_list]
    poi_ids = list(set(poi_ids_flat))
    poi_counts = [len([el for el in poi_ids_flat if el == poi]) for poi in list(set(poi_ids))]
    pois_ids_min_count = [poi_ids_flat[j] for j in range(len(poi_ids)) if poi_counts[j] >= min_count]
    pois = rt.Route([route[idx] for idx in pois_ids_min_count])
    return pois, pois_ids_min_count
