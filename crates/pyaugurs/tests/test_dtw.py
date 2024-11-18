import numpy as np
import pytest

from augurs import Dtw


class TestDtwInstantiation:
    """Test DTW instantiation."""

    def test_default_instantiation(self):
        """Test DTW can be instantiated with defaults."""
        dtw = Dtw()
        assert dtw is not None
        assert isinstance(dtw, Dtw)

    def test_instantiation_with_euclidean(self):
        """Test instantiation with euclidean distance."""
        dtw = Dtw(distance_fn="euclidean")
        assert dtw is not None

    def test_instantiation_with_manhattan(self):
        """Test instantiation with manhattan distance."""
        dtw = Dtw(distance_fn="manhattan")
        assert dtw is not None

    def test_instantiation_with_window(self):
        """Test instantiation with custom window."""
        dtw = Dtw(window=5)
        assert dtw is not None

    def test_instantiation_with_all_parameters(self):
        """Test instantiation with all parameters."""
        dtw = Dtw(window=10, distance_fn="manhattan")
        assert dtw is not None


class TestDtw:
    @pytest.mark.parametrize(
        "opts, input, expected",
        [
            ({}, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 5.0990195135926845),
            ({"window": 2}, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 5.0990195135926845),
            (
                {"distance_fn": "euclidean"},
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                5.0990195135926845,
            ),
            ({"distance_fn": "manhattan"}, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 9),
        ],
    )
    def test_distance(self, opts, input, expected):
        d = Dtw(**opts)
        arrays = [np.array(x) for x in input]
        np.testing.assert_allclose(d.distance(*arrays), expected)

    @pytest.mark.parametrize(
        "opts, input, expected",
        [
            (
                {},
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [
                    [0.0, 5.0990195135926845, 10.392304845413264],
                    [5.0990195135926845, 0.0, 5.0990195135926845],
                    [10.392304845413264, 5.0990195135926845, 0.0],
                ],
            ),
            (
                {"distance_fn": "manhattan"},
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [
                    [0.0, 9.0, 18.0],
                    [9.0, 0.0, 9.0],
                    [18.0, 9.0, 0.0],
                ],
            ),
        ],
    )
    def test_distance_matrix(self, opts, input, expected):
        d = Dtw(**opts)
        arrays = [np.array(x) for x in input]
        np.testing.assert_allclose(d.distance_matrix(arrays).to_numpy(), expected)

    def test_distance_with_typed_arrays(self):
        """Test distance calculation with typed numpy arrays."""
        dtw = Dtw(distance_fn="euclidean")
        a = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0, 5.0], dtype=np.float64)
        distance = dtw.distance(a, b)
        np.testing.assert_allclose(distance, 5.0990195135926845)

    def test_distance_with_different_lengths(self):
        """Test distance with arrays of different lengths."""
        dtw = Dtw(distance_fn="euclidean")
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        distance = dtw.distance(a, b)
        # Should handle different lengths
        assert distance > 0

    def test_distance_with_identical_arrays(self):
        """Test distance between identical arrays is zero."""
        dtw = Dtw(distance_fn="euclidean")
        a = np.array([1.0, 2.0, 3.0, 4.0])
        distance = dtw.distance(a, a)
        np.testing.assert_allclose(distance, 0.0)

    def test_distance_with_empty_arrays(self):
        """Test distance with empty arrays."""
        dtw = Dtw(distance_fn="euclidean")
        a = np.array([])
        b = np.array([3.0, 4.0, 5.0])
        # Empty array should result in infinite distance
        distance = dtw.distance(a, b)
        assert np.isinf(distance)

    def test_distance_both_empty(self):
        """Test distance with both arrays empty."""
        dtw = Dtw(distance_fn="euclidean")
        a = np.array([])
        b = np.array([])
        distance = dtw.distance(a, b)
        assert np.isinf(distance)

    def test_manhattan_distance_specific(self):
        """Test manhattan distance with specific data."""
        dtw = Dtw(distance_fn="manhattan")
        a = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        distance = dtw.distance(a, b)
        np.testing.assert_allclose(distance, 2.0)

    def test_distance_with_single_element(self):
        """Test distance with single element arrays."""
        dtw = Dtw(distance_fn="euclidean")
        a = np.array([1.0])
        b = np.array([4.0])
        distance = dtw.distance(a, b)
        np.testing.assert_allclose(distance, 3.0)

    def test_distance_with_negative_values(self):
        """Test distance with negative values."""
        dtw = Dtw(distance_fn="euclidean")
        a = np.array([-1.0, -2.0, -3.0])
        b = np.array([1.0, 2.0, 3.0])
        distance = dtw.distance(a, b)
        assert distance > 0

    def test_distance_with_larger_window(self):
        """Test distance with larger window parameter."""
        dtw = Dtw(window=10, distance_fn="euclidean")
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        distance = dtw.distance(a, b)
        assert distance > 0

    def test_distance_matrix_with_typed_arrays(self):
        """Test distance matrix with typed arrays."""
        dtw = Dtw(distance_fn="euclidean")
        series = [
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0, 5.0], dtype=np.float64),
            np.array([6.0, 7.0, 8.0], dtype=np.float64),
        ]
        dists = dtw.distance_matrix(series)
        matrix = dists.to_numpy()

        assert matrix.shape == (3, 3)
        # Diagonal should be zero
        np.testing.assert_allclose(matrix[0][0], 0.0)
        np.testing.assert_allclose(matrix[1][1], 0.0)
        np.testing.assert_allclose(matrix[2][2], 0.0)
        # Should be symmetric
        np.testing.assert_allclose(matrix[0][1], matrix[1][0])

    def test_distance_matrix_symmetry(self):
        """Test that distance matrix is symmetric."""
        dtw = Dtw(distance_fn="euclidean")
        series = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]
        dists = dtw.distance_matrix(series)
        matrix = dists.to_numpy()

        # Check symmetry
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(matrix[i][j], matrix[j][i])

    def test_distance_matrix_single_series(self):
        """Test distance matrix with single series."""
        dtw = Dtw(distance_fn="euclidean")
        series = [np.array([1.0, 2.0, 3.0])]
        dists = dtw.distance_matrix(series)
        matrix = dists.to_numpy()

        assert matrix.shape == (1, 1)
        np.testing.assert_allclose(matrix[0][0], 0.0)

    def test_distance_matrix_two_series(self):
        """Test distance matrix with two series."""
        dtw = Dtw(distance_fn="euclidean")
        series = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        dists = dtw.distance_matrix(series)
        matrix = dists.to_numpy()

        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix[0][0], 0.0)
        np.testing.assert_allclose(matrix[1][1], 0.0)
        assert matrix[0][1] > 0
        np.testing.assert_allclose(matrix[0][1], matrix[1][0])

    def test_distance_matrix_with_different_lengths(self):
        """Test distance matrix with series of different lengths."""
        dtw = Dtw(distance_fn="euclidean")
        series = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
            np.array([6.0, 7.0, 8.0, 9.0]),
        ]
        dists = dtw.distance_matrix(series)
        matrix = dists.to_numpy()

        assert matrix.shape == (3, 3)
        # Diagonal should be zero
        for i in range(3):
            np.testing.assert_allclose(matrix[i][i], 0.0)

    def test_distance_matrix_manhattan(self):
        """Test distance matrix with Manhattan distance."""
        dtw = Dtw(distance_fn="manhattan")
        series = [
            np.array([0.0, 1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
        ]
        dists = dtw.distance_matrix(series)
        matrix = dists.to_numpy()

        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix[0][0], 0.0)
        np.testing.assert_allclose(matrix[1][1], 0.0)
        np.testing.assert_allclose(matrix[0][1], 9.0)

    def test_readme_example(self):
        """Test the example from the README."""
        dtw = Dtw()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        # Distance calculation
        dist = dtw.distance(a, b)
        assert dist > 0

        # Distance matrix calculation
        dist_matrix = dtw.distance_matrix([a, b])
        assert dist_matrix is not None
        matrix = dist_matrix.to_numpy()
        assert matrix.shape == (2, 2)

    def test_reuse_dtw_instance(self):
        """Test that DTW instance can be reused for multiple calculations."""
        dtw = Dtw(distance_fn="euclidean")

        # First calculation
        a1 = np.array([1.0, 2.0, 3.0])
        b1 = np.array([4.0, 5.0, 6.0])
        dist1 = dtw.distance(a1, b1)

        # Second calculation
        a2 = np.array([0.0, 0.0, 0.0])
        b2 = np.array([1.0, 1.0, 1.0])
        dist2 = dtw.distance(a2, b2)

        # Both should work and give different results
        assert dist1 > 0
        assert dist2 > 0
        assert dist1 != dist2

    def test_distance_with_real_time_series(self):
        """Test with realistic time series data."""
        dtw = Dtw(distance_fn="euclidean")

        # Generate two sinusoidal patterns with slight phase shift
        t1 = np.linspace(0, 4 * np.pi, 50)
        series1 = np.sin(t1)
        series2 = np.sin(t1 + 0.1)  # Slight phase shift

        distance = dtw.distance(series1, series2)
        # Should be relatively small since patterns are similar
        assert distance > 0
        assert distance < 10  # Arbitrary reasonable upper bound

    def test_distance_matrix_large(self):
        """Test distance matrix with more series."""
        dtw = Dtw(distance_fn="euclidean")

        # Create 5 series
        series = [np.array([float(i), float(i + 1), float(i + 2)]) for i in range(5)]
        dists = dtw.distance_matrix(series)
        matrix = dists.to_numpy()

        assert matrix.shape == (5, 5)
        # Check diagonal is zero
        for i in range(5):
            np.testing.assert_allclose(matrix[i][i], 0.0)
        # Check symmetry
        for i in range(5):
            for j in range(5):
                np.testing.assert_allclose(matrix[i][j], matrix[j][i])
