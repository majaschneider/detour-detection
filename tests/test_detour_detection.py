"""Test of detour detection.
"""

from math import radians, cos, sin
import unittest
from datetime import datetime as dt

import openrouteservice
import pandas as pd
from geodata.geodatasets import de4l
from geodata.geodata import route as rt
from geodata.geodata import point as pt
from geopy.geocoders import Nominatim
from geodata.helper import parser

from detour_detection import detour_detection


# todo: move to geodata
def convert_stops(stops):
    return [int(s) for s in stops.replace('[', '').replace(']', '').split(', ')]


def convert_timestamps(timestamps):
    return [pd.Timestamp(timestamp) for timestamp in
            timestamps.replace('[', '').replace(']', '').replace("Timestamp('", '').replace("')", '').split(', ')]


def move_(coordinates, x_delta, y_delta):
    """
    Moves the coordinates tuple in x and y direction. The coordinates object is changed in place and returned.

    Parameters
    ----------
    coordinates : List
        A list of two float coordinates x and y.
    x_delta : float
        The distance which should be added onto the x-coordinate.
    y_delta : float
        The distance which should be added onto the y-coordinate.

    Returns
    -------
    coordinates : List
        The coordinates increased by x_delta and y_delta.
    """
    coordinates[0] += x_delta
    coordinates[1] += y_delta
    return coordinates


def set_(coordinates, new_coordinates):
    """
    Set the coordinates tuple to new coordinate values. The coordinates object is changed in place and returned.

    Parameters
    ----------
    coordinates : List
        A list of two float coordinates x and y.
    new_coordinates : List
        A list of the two new float coordinates x and y.

    Returns
    -------
    coordinates : List
        The coordinates object whose values are changed to the provided new values.
    """
    coordinates[0] = new_coordinates[0]
    coordinates[1] = new_coordinates[1]
    return coordinates


def get_performance(true_stops, detected_stops, max_distance):
    """
    Evaluate the performance of a stop/poi detection algorithm by calculating the number of true-positives,
    false-positives and false-negatives where:
        true-positives : Points that have correctly been identified as a stop. A point from detected_stops is correctly
        identified as a true stop, when it is less than max_distance apart from any true stop.

        false-positives : Points that have incorrectly been identified as a stop.

        false-negatives : Stops that have not been detected.

    Detected stops are assigned to their closest true stop and counted as true-positive, only if the true stop has not
    been assigned to any other detected stop before. There might be rare edge-cases where a detected stop could be
    assigned to multiple true stops, e.g. stop1 and stop2 with stop1 being closer, and by assigning the detected stop
    not to the closest true stop (stop1) but to stop2, another detected stop could be assigned to stop1 and counted
    as true-positive, whereas in the current implementation it is ignored. This is an alignment problem but should not
    be relevant in practice, as we assume, that a detected stop should naturally be assigned to its closest true stop.

    Parameters
    ----------
    true_stops : Route
        The list of true stops/pois.
    detected_stops : Route
        The list of stops/pois that have been detected by an algorithm.
    max_distance : float
        The maximum distance in meters that a point is allowed to be apart from a true stop to be counted as such.

    Returns
    -------
    (true-positives, false-positives, false-negatives) : (int, int, int)
        The performance of the detection algorithm denoted by the number of true-positives, false-positives and
         false-negatives.
    """
    true_positives = 0
    false_positives = 0

    # for each detected stop find the closest true stop that is not further apart than max_distance
    found_true_stops = []
    number_of_detected_stops = len(detected_stops)
    for idx in range(number_of_detected_stops):
        detected_stop = detected_stops[idx]
        min_distance = None
        true_stop_index = None
        for true_stop in true_stops:
            distance = pt.get_distance(detected_stop, true_stop)
            if distance <= max_distance and (min_distance is None or min_distance > distance):
                min_distance = distance
                true_stop_index = true_stops.index(true_stop)
        # if no true stop is found
        if true_stop_index is None:
            false_positives += 1
        # if a close enough true stop is found
        else:
            # if the found true stop is already assigned to a detected stop, ignore the current detected stop even if
            # it might be closer to the true stop
            if true_stop_index not in found_true_stops:
                true_positives += 1
                found_true_stops.append(true_stop_index)

    false_negatives = len(true_stops) - len(found_true_stops)

    return true_positives, false_positives, false_negatives


class TestMetric(unittest.TestCase):
    """Test of detour detection.
    """

    def test_select_samples_by_temporal_distance(self):
        """Test of sample selection by temporal distance from a route.
        """
        # Manually define the timestamps that should be in the test dataset
        timestamps = [
            pd.Timestamp('2021-02-22T10:31:33.000Z'),
            pd.Timestamp('2021-02-22T10:31:52.000Z'),
            pd.Timestamp('2021-02-22T10:32:07.000Z'),
            pd.Timestamp('2021-02-22T10:32:17.000Z'),
            pd.Timestamp('2021-02-22T10:32:40.000Z'),
            pd.Timestamp('2021-02-22T10:32:41.000Z'),
            pd.Timestamp('2021-02-22T10:32:50.000Z'),
            pd.Timestamp('2021-02-22T10:33:00.000Z'),
            pd.Timestamp('2021-02-22T10:33:23.000Z'),
            pd.Timestamp('2021-02-22T10:33:26.000Z'),
            pd.Timestamp('2021-02-22T10:33:27.000Z')
        ]

        # Define desired timedeltas for the iteration
        timedeltas = [
            pd.Timedelta(value=0, unit='sec'),
            pd.Timedelta(value=1, unit='sec'),
            pd.Timedelta(value=2, unit='sec'),
            pd.Timedelta(value=5, unit='sec'),
            pd.Timedelta(value=15, unit='sec'),
            pd.Timedelta(value=30, unit='sec'),
            pd.Timedelta(value=1, unit='min'),
            pd.Timedelta(value=2, unit='min'),
            pd.Timedelta(value=1500, unit='millis'),
        ]

        dataset = de4l.De4lSensorDataset.create_from_json('resources/test-dataset.json', route_len=11)

        # Get a route from the dataset
        route = dataset[0]['route_with_timestamps']

        # Define different timestamps to test sampling with start timestamp
        start_timestamp_middle = pd.Timestamp('2021-02-22T10:32:41.000Z')
        start_timestamp_late = pd.Timestamp('2021-02-22T10:33:27.000Z')
        start_timestamp_too_late = pd.Timestamp('2021-02-22T10:33:28.000Z')

        # Test different parameter setups that contain temporal distance, start timestamp and indices of expected
        # sampled points and validate the sampled points
        for temporal_distance, start_timestamp, expected_indices in [
            # Without start timestamp
            (timedeltas[0], None, list(range(11))),
            (timedeltas[1], None, list(range(11))),
            (timedeltas[2], None, [0, 1, 2, 3, 4, 6, 7, 8, 9]),
            (timedeltas[3], None, [0, 1, 2, 3, 4, 6, 7, 8]),
            (timedeltas[4], None, [0, 1, 2, 4, 7, 8]),
            (timedeltas[5], None, [0, 2, 4, 8]),
            (timedeltas[6], None, [0, 4]),
            (timedeltas[7], None, [0]),
            (timedeltas[8], None, [0, 1, 2, 3, 4, 6, 7, 8, 9]),
            # With start timestamp in the middle of the timestamps to sample
            (timedeltas[1], start_timestamp_middle, list(range(5, 11))),
            (timedeltas[2], start_timestamp_middle, [5, 6, 7, 8, 9]),
            (timedeltas[3], start_timestamp_middle, [5, 6, 7, 8]),
            (timedeltas[4], start_timestamp_middle, [5, 7, 8]),
            (timedeltas[5], start_timestamp_middle, [5, 8]),
            (timedeltas[6], start_timestamp_middle, [5]),
            (timedeltas[7], start_timestamp_middle, [5]),
            # With start timestamp at the end of the timestamps to sample
            (timedeltas[3], start_timestamp_late, [10]),
            # With start timestamp later than the timestamps to sample
            (timedeltas[3], start_timestamp_too_late, []),
        ]:
            _, sampled_points = detour_detection.select_samples_by_temporal_distance(route, temporal_distance,
                                                                                     start_timestamp)
            expected_timestamps = [timestamps[idx] for idx in expected_indices]
            sampled_timestamps = [point.timestamp for point in sampled_points]
            self.assertEqual(expected_timestamps, sampled_timestamps)

        # Test parameters that should lead to an error
        with self.assertRaises(ValueError):
            detour_detection.select_samples_by_temporal_distance(route, temporal_distance='1')
            detour_detection.select_samples_by_temporal_distance(route, temporal_distance=1)
            detour_detection.select_samples_by_temporal_distance(route, temporal_distance=timedeltas[1],
                                                                 start_timestamp='2021-02-22T10:32:41.000Z')
            detour_detection.select_samples_by_temporal_distance(route, temporal_distance=timedeltas[1],
                                                                 start_timestamp=1613989893.0)

            # Disallow timestamps or timedeltas of the datetime module
            detour_detection.select_samples_by_temporal_distance(route, temporal_distance=dt.timedelta(seconds=5),
                                                                 start_timestamp=timestamps[0])
            detour_detection.select_samples_by_temporal_distance(route, timedeltas[1], dt.datetime.
                                                                 fromtimestamp('2021-02-22T10:32:41.000Z'))

            # Test if passing an empty route leads to an error
            detour_detection.select_samples_by_temporal_distance(rt.Route(), temporal_distance=timedeltas[1])

            # Test if passing a wrong type for route leads to an error
            detour_detection.select_samples_by_temporal_distance([[1, 1]], temporal_distance=timedeltas[1])

            # Test if passing a route containing points without timestamps leads to an error
            route_without_timestamps = rt.Route()
            route_without_timestamps.append(pt.Point([51.3396955, 12.3730747]))
            route_without_timestamps.append(pt.Point([51.0504088, 13.7372621]))
            detour_detection.select_samples_by_temporal_distance(route_without_timestamps,
                                                                 temporal_distance=timedeltas[1])

    def test_select_samples_by_spatial_distance(self):
        """Test of sample selection by spatial distance from a route.
        """
        # design the example route
        #                         |
        #                         |
        #                         |
        #             ____________|
        #            /
        # __________/
        # start at coordinate origin
        point1 = pt.Point([0, 0], geo_reference_system='cartesian')
        point1.to_latlon_()
        # move along x-axis for 100 meters
        point2 = point1.add_vector(100, radians(0))
        # move in 45 degrees angle north east for 50 meters
        point3 = point2.add_vector(50, radians(45))
        # move 100 meters east parallel to the x-axis
        point4 = point3.add_vector(100, radians(0))
        # move 100 meters north parallel to the y-axis
        point5 = point4.add_vector(100, radians(90))

        route = rt.Route([point1, point2, point3, point4, point5])
        _, selected_samples = detour_detection.select_samples_by_spatial_distance(route, spatial_distance=125.0)
        expected_result = rt.Route([point1, point3, point5])

        self.assertEqual(expected_result, selected_samples)

    def test_sample_from_shape(self):
        """Test of sample generation from the shape of a route.
        """
        # design the example shape
        # start at coordinate origin
        point1 = pt.Point([0, 0])
        angle1 = radians(0)
        # move along x-axis for 100 meters
        point2 = point1.add_vector(100, angle1)
        angle2 = radians(45)
        # move in 45 degrees angle north east for 50 meters
        point3 = point2.add_vector(50, angle2)
        example_shape = rt.Route([point1, point2, point3])

        # When sampling every 20 meters, we expect to sample six points along the x-axis with a distance between each
        # of 20 meters ([0, 0] up to [0, 100 meters]) and then two points in a 45 degrees angle north east, also with a
        # distance of 20 meters between each point. In a rectangular triangle trigonometric equations can be used to
        # determine these last two points:
        #
        #      C
        #     /|
        #    / |
        #   /  |
        #  /   |
        # A----B
        # angle at A is 45°
        # angle at B is 90°
        #
        # calculate y-elevation:
        # sin(45°) = opposite leg/hypotenuse = |BC|/|AC|
        # |AC| = 20 -> |BC| = sin(45°) * 20
        #
        # calculate x-elevation:
        # cos(45°) = opposite leg/hypotenuse = |AB|/|AC|
        # |AC| = 20 -> |AB| = cos(45°) * 20
        sampling_distance = 20
        meters_factor = 0.001  # all points are located in a cartesian plane of which each axis is measured in
        # kilometers
        expected_sampled_points = rt.Route(
            [pt.Point([0, 0], 'cartesian'),
             pt.Point([0, 20 * meters_factor], 'cartesian'),
             pt.Point([0, 40 * meters_factor], 'cartesian'),
             pt.Point([0, 60 * meters_factor], 'cartesian'),
             pt.Point([0, 80 * meters_factor], 'cartesian'),
             pt.Point([0, 100 * meters_factor], 'cartesian'),
             pt.Point([cos(angle2) * 20 * meters_factor,
                       100 * meters_factor + sin(angle2) * 20 * meters_factor], 'cartesian'),
             pt.Point([cos(angle2) * 40 * meters_factor,
                       100 * meters_factor + sin(angle2) * 40 * meters_factor], 'cartesian')])

        sampled_points = detour_detection.sample_from_shape(example_shape, sampling_distance)
        len_sampled_points = len(sampled_points)

        self.assertEqual(len(expected_sampled_points), len_sampled_points)

        for idx in range(len_sampled_points):
            sampled_points[idx].to_cartesian_()
            self.assertAlmostEqual(expected_sampled_points[idx].x_lon, sampled_points[idx].x_lon, 10)
            self.assertAlmostEqual(expected_sampled_points[idx].y_lat, sampled_points[idx].y_lat, 10)

    def test_reverse_geocode(self):
        """Test of reverse geocoding of a list of geographical points to addresses.
        """
        # todo: test reverse geocoding, once an instance of Nominatim is available from GitLab
        # nominatim_url = 'localhost:8080'
        # route = rt.Route(pt.Point([1, 1], coordinates_unit='radians'))
        #
        # reversed_route, failed_requests = detour_detection.reverse_geocode(route, nominatim_url)
        # # Expect no failed requests
        # self.assertEqual(rt.Route, reversed_route)
        # self.assertEqual(0, failed_requests)

    def test_nominatim_reverse(self):
        """Test of reverse geocoding of a geographical point to an address.
        """
        nominatim_url = 'localhost:8080'
        nominatim = Nominatim(scheme='http', domain=nominatim_url)

        to_reverse = [(pt.Point([2.439017920469, 0.62235962758]), ValueError)]    # Tokyo Tower Tokyo, real coordinates

        # There should be value errors for every entry and different logs about the individual errors
        for point, expected_error in to_reverse:
            self.assertRaises(expected_error, detour_detection.nominatim_reverse, point, nominatim)

    def test_get_directions_for_points(self):
        """Test of the shortest route calculation between two geographical points.
        """
        # todo: test ors services, once an instance of ORS is available from GitLab
        start_point = pt.Point([0.9, 0.1])
        end_point = pt.Point([0.8, 0.2])

        # Raise because of wrong point parameters
        for wrong_point in [[], [1.0, 1.0], '[1.0, 1.0]']:
            self.assertRaises(ValueError,
                              detour_detection.get_directions_for_points,
                              start=wrong_point,
                              end=end_point,
                              openrouteservice_client=openrouteservice.Client(key=""))

        # Raise because of missing openrouteservice client
        self.assertRaises(ValueError,
                          detour_detection.get_directions_for_points,
                          start=start_point,
                          end=end_point,
                          openrouteservice_client=None,
                          openrouteservice_profile='driving-car')

        # Raise because of wrong openrouteservice profile
        self.assertRaises(ValueError,
                          detour_detection.get_directions_for_points,
                          start=start_point,
                          end=end_point,
                          openrouteservice_client=openrouteservice.Client(key=""),
                          openrouteservice_profile='rocket_spaceship')

    def test_get_directions_for_route(self):
        """Test of the shortest route calculation between each two geographical points of a route.
        """
        correct_route = rt.Route([pt.Point([-8.629335172276479, 41.15916914599747], coordinates_unit='degrees'),
                                  pt.Point([-8.659118422444477, 41.16278779752667], coordinates_unit='degrees')])
        base_path = 'localhost:8008'

        # todo: test ors services, once an instance of ORS is available from GitLab
        # Successfully calculate shortest route details
        # detour_detection.get_directions_for_route(correct_route, base_path)

        # Raise because of an invalid route
        for wrong_route in [[], rt.Route(), rt.Route([pt.Point([0.4, 0.9])])]:
            self.assertRaises(ValueError,
                              detour_detection.get_directions_for_route,
                              route=wrong_route,
                              openrouteservice_base_path=base_path)

        # Raise because of a wrong openrouteservice profile
        self.assertRaises(ValueError,
                          detour_detection.get_directions_for_route,
                          route=correct_route,
                          openrouteservice_base_path=base_path,
                          openrouteservice_profile='rocket_spaceship')

        # Warn in case of a broken connection
        for _ in ['localhost:9008', 'localhos:8008', 'localhost']:
            self.assertWarns(UserWarning)

    def test_detect_pois_from_exceedings(self):
        for exceedings, expected_pois in [
            ([0, 1, 0, 2, 5], [1, 4]),
            ([1, 2, 0, 5, 2], [1, 3]),
            ([1, 0, 0, 2, 5], [0, 4]),
            ([0, 1, 0, 0, 0], [1]),
            ([1, 1, 0, 0, 0], [0]),
            ([1], [0]),
            ([0], [])
        ]:
            self.assertEqual(expected_pois, detour_detection.detect_pois_from_exceedings(exceedings))

    def test_calculate_common_poi_risk(self):
        ors_path = '172.17.2.117:50003'
        nominatim_url = '172.17.2.117:50002'
        scheme = 'https'
        ors_profile = 'driving-car'
        nominatim = Nominatim(scheme=scheme, domain=nominatim_url)
        max_distance = 200

        file_path = 'resources/test_porto_taxi_stop_based.csv'
        df = pd.read_csv(file_path, sep=',', encoding='latin1')
        idx_route = 32
        row = df.iloc[idx_route]
        timestamps = convert_timestamps(row['timestamps'])
        route = rt.Route(parser.route_str_to_list(row['route']), timestamps)
        expected_pois = rt.Route([route[i] for i in convert_stops(row['stops'])])
        # remove start and end points
        expected_pois = rt.Route(expected_pois[1:-1])

        parameters = [
            (100, 10, 20),
            (310, 10, 20),
            (450, 10, 20)
        ]
        pois, _ = detour_detection.calculate_common_poi_risk(parameters=parameters,
                                                             min_count=len(parameters),
                                                             route=route,
                                                             ors_path=ors_path,
                                                             ors_scheme=scheme,
                                                             ors_profile=ors_profile,
                                                             nominatim=nominatim,
                                                             map_matching=False)

        true_positives, false_positives, false_negatives = get_performance(expected_pois, pois, max_distance)
        print(true_positives, false_positives, false_negatives)
        # doesn't recognize all but at least one POI
        self.assertEqual(0, false_positives)
        self.assertEqual(1, true_positives)

    def test_calculate_poi_risk(self):
        ors_path = '172.17.2.117:50003'
        nominatim_url = '172.17.2.117:50002'
        scheme = 'https'
        ors_profile = 'driving-car'
        nominatim = Nominatim(scheme=scheme, domain=nominatim_url)
        max_distance = 200

        file_path = 'resources/test_porto_taxi_stop_based.csv'
        data_frame = pd.read_csv(file_path, sep=',', encoding='latin1')
        idx_route = 32
        row = data_frame.iloc[idx_route]
        timestamps = convert_timestamps(row['timestamps'])
        route = rt.Route(parser.route_str_to_list(row['route']), timestamps)
        expected_pois = rt.Route([route[i] for i in convert_stops(row['stops'])])
        # remove start and end points from expectation
        expected_pois = rt.Route(expected_pois[1:-1])

        _, _, _, _, _, detected_pois, _ = \
            detour_detection.calculate_poi_risk(route=route,
                                                spatial_distance=310,
                                                ors_path=ors_path,
                                                ors_scheme=scheme,
                                                ors_profile=ors_profile,
                                                nominatim=nominatim,
                                                sampling_distance=10,
                                                acceptable_distance_to_shortest_route=20,
                                                map_matching=False)
        true_positives, false_positives, false_negatives = get_performance(expected_pois, detected_pois, max_distance)
        self.assertEqual(1, true_positives)


if __name__ == '__main__':
    unittest.main()
