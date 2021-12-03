import unittest
import datetime

import pandas as pd
from de4l_geodata.geodatasets import de4l
from de4l_geodata.geodata import route as rt
from de4l_geodata.geodata import point as pt

from de4l_detour_detection import detour_detection


class TestMetric(unittest.TestCase):

    def test_select_samples(self):

        # Manually define the timestamps that should be in the test dataset
        timestamps = [
            pd.Timestamp("2021-02-22T10:31:33.000Z"),
            pd.Timestamp("2021-02-22T10:31:52.000Z"),
            pd.Timestamp("2021-02-22T10:32:07.000Z"),
            pd.Timestamp("2021-02-22T10:32:17.000Z"),
            pd.Timestamp("2021-02-22T10:32:40.000Z"),
            pd.Timestamp("2021-02-22T10:32:41.000Z"),
            pd.Timestamp("2021-02-22T10:32:50.000Z"),
            pd.Timestamp("2021-02-22T10:33:00.000Z"),
            pd.Timestamp("2021-02-22T10:33:23.000Z"),
            pd.Timestamp("2021-02-22T10:33:26.000Z"),
            pd.Timestamp("2021-02-22T10:33:27.000Z")
        ]

        # Define desired timedeltas for the iteration
        timedeltas = [
            pd.Timedelta(value=0, unit="sec"),
            pd.Timedelta(value=1, unit="sec"),
            pd.Timedelta(value=2, unit="sec"),
            pd.Timedelta(value=5, unit="sec"),
            pd.Timedelta(value=15, unit="sec"),
            pd.Timedelta(value=30, unit="sec"),
            pd.Timedelta(value=1, unit="min"),
            pd.Timedelta(value=2, unit="min"),
            pd.Timedelta(value=1500, unit="millis"),
        ]

        dataset = de4l.De4lSensorDataset.create_from_json("resources/test-dataset.json", route_len=11)

        # Get a route from the dataset
        route = dataset[0]["route_with_timestamps"]

        # Define different timestamps to test sampling with start timestamp
        start_timestamp_middle = pd.Timestamp("2021-02-22T10:32:41.000Z")
        start_timestamp_late = pd.Timestamp("2021-02-22T10:33:27.000Z")
        start_timestamp_too_late = pd.Timestamp("2021-02-22T10:33:28.000Z")

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
            sampled_points = detour_detection.select_samples(route, temporal_distance, start_timestamp)
            expected_timestamps = [timestamps[idx] for idx in expected_indices]
            sampled_timestamps = [point.timestamp for point in sampled_points]
            self.assertEqual(expected_timestamps, sampled_timestamps)

        # Test parameters that should lead to an error
        with self.assertRaises(ValueError):
            detour_detection.select_samples(route, temporal_distance="1")
            detour_detection.select_samples(route, temporal_distance=1)
            detour_detection.select_samples(route,
                                            temporal_distance=timedeltas[1],
                                            start_timestamp="2021-02-22T10:32:41.000Z")
            detour_detection.select_samples(route,
                                            temporal_distance=timedeltas[1],
                                            start_timestamp=1613989893.0)

            # Disallow timestamps or timedeltas of the datetime module
            detour_detection.select_samples(route,
                                            temporal_distance=datetime.timedelta(seconds=5),
                                            start_timestamp=timestamps[0])
            detour_detection.select_samples(route,
                                            timedeltas[1],
                                            datetime.datetime.fromtimestamp("2021-02-22T10:32:41.000Z"))

            # Test if passing an empty route leads to an error
            detour_detection.select_samples(rt.Route(), temporal_distance=timedeltas[1])

            # Test if passing a wrong type for route leads to an error
            detour_detection.select_samples([[1, 1]], temporal_distance=timedeltas[1])

            # Test if passing a route containing points without timestamps leads to an error
            route_without_timestamps = rt.Route()
            route_without_timestamps.append(pt.Point([51.3396955, 12.3730747]))
            route_without_timestamps.append(pt.Point([51.0504088, 13.7372621]))
            detour_detection.select_samples(route_without_timestamps, temporal_distance=timedeltas[1])

    def test_reverse_geocode(self):

        nominatim_url = "localhost:1234"

        to_reverse = [
            pt.Point([0.62235962758, 2.439017920469]),  # Tokyo Tower Tokyo, real coordinates
            pt.Point([3.62235962758, 4.439017920469]),  # Radians values too high, bogus coordinates
            # The following values are not instances of geodata.point.Point
            [0.9166228376, 0.233458504512],
            ["0.9166228376", "0.233458504512"],
            123,
            "456"
        ]

        reversed_route, failed_requests, wrong_values = detour_detection.reverse_geocode(to_reverse, nominatim_url)
        # We expect one wrong value and two failed requests
        self.assertEqual([], reversed_route)
        self.assertEqual(2, failed_requests)
        self.assertEqual(4, wrong_values)


if __name__ == "__main__":
    unittest.main()