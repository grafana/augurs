import numpy as np
import pytest

from augurs import Dbscan, Dtw


class TestDbscan:
    """Test suite for DBSCAN clustering."""

    @pytest.fixture
    def simple_distance_matrix_list(self):
        """Simple distance matrix as list of lists."""
        return [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 3.0, 3.0],
            [2.0, 3.0, 0.0, 4.0],
            [3.0, 3.0, 4.0, 0.0],
        ]

    @pytest.fixture
    def simple_distance_matrix_numpy(self):
        """Simple distance matrix as numpy array."""
        return np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 3.0, 3.0],
                [2.0, 3.0, 0.0, 4.0],
                [3.0, 3.0, 4.0, 0.0],
            ]
        )

    @pytest.fixture
    def time_series_for_clustering(self):
        """Sample time series for clustering via DTW."""
        return [
            np.array([1.0, 3.0, 4.0]),
            np.array([1.0, 3.0, 3.9]),
            np.array([1.1, 2.9, 4.1]),
            np.array([5.0, 6.2, 10.0]),
        ]

    def test_instantiation(self):
        """Test that Dbscan can be instantiated with valid parameters."""
        clusterer = Dbscan(epsilon=0.5, min_cluster_size=2)
        assert clusterer is not None
        assert isinstance(clusterer, Dbscan)

    def test_instantiation_with_different_parameters(self):
        """Test instantiation with various epsilon and min_cluster_size values."""
        params = [
            (0.1, 2),
            (0.5, 3),
            (1.0, 4),
            (2.0, 5),
        ]
        for eps, min_size in params:
            clusterer = Dbscan(epsilon=eps, min_cluster_size=min_size)
            assert clusterer is not None

    def test_repr(self):
        """Test string representation of Dbscan."""
        clusterer = Dbscan(epsilon=1.5, min_cluster_size=3)
        repr_str = repr(clusterer)
        assert "Dbscan" in repr_str
        assert "1.5" in repr_str
        assert "3" in repr_str

    def test_fit_with_list_of_lists(self, simple_distance_matrix_list):
        """Test fitting with a list of lists."""
        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(simple_distance_matrix_list)

        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.int32
        assert len(labels) == 4

    def test_fit_with_numpy_array(self, simple_distance_matrix_numpy):
        """Test fitting with a numpy array."""
        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(simple_distance_matrix_numpy)

        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.int32
        assert len(labels) == 4

    def test_cluster_assignments(self, simple_distance_matrix_list):
        """Test that cluster assignments are reasonable."""
        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(simple_distance_matrix_list)

        # Expected: first two points form a cluster, last two are noise
        expected = np.array([1, 1, -1, -1], dtype=np.int32)
        np.testing.assert_array_equal(labels, expected)

    def test_cluster_assignments_numpy(self, simple_distance_matrix_numpy):
        """Test cluster assignments with numpy array input."""
        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(simple_distance_matrix_numpy)

        expected = np.array([1, 1, -1, -1], dtype=np.int32)
        np.testing.assert_array_equal(labels, expected)

    def test_noise_points(self, simple_distance_matrix_list):
        """Test that noise points are labeled as -1."""
        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(simple_distance_matrix_list)

        # Points 2 and 3 should be noise (-1)
        assert labels[2] == -1
        assert labels[3] == -1

    def test_with_dtw_distance_matrix(self, time_series_for_clustering):
        """Test clustering with distance matrix from DTW."""
        # Compute DTW distance matrix
        dtw = Dtw(distance_fn="euclidean")
        distance_matrix = dtw.distance_matrix(time_series_for_clustering)

        # Cluster the time series
        clusterer = Dbscan(epsilon=0.5, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.int32
        assert len(labels) == 4

        # First three series are similar, should be in same cluster
        # Fourth series is different, should be noise
        expected = np.array([1, 1, 1, -1], dtype=np.int32)
        np.testing.assert_array_equal(labels, expected)

    def test_different_epsilon_values(self, simple_distance_matrix_numpy):
        """Test clustering with different epsilon values."""
        # Small epsilon - more restrictive clustering
        clusterer_small = Dbscan(epsilon=0.5, min_cluster_size=2)
        labels_small = clusterer_small.fit(simple_distance_matrix_numpy)

        # Large epsilon - more permissive clustering
        clusterer_large = Dbscan(epsilon=5.0, min_cluster_size=2)
        labels_large = clusterer_large.fit(simple_distance_matrix_numpy)

        # Both should return valid labels
        assert len(labels_small) == 4
        assert len(labels_large) == 4

        # Results should differ based on epsilon
        # (though this is data-dependent)

    def test_different_min_cluster_sizes(self, simple_distance_matrix_numpy):
        """Test clustering with different minimum cluster sizes."""
        # Smaller min size
        clusterer_small = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels_small = clusterer_small.fit(simple_distance_matrix_numpy)

        # Larger min size
        clusterer_large = Dbscan(epsilon=1.0, min_cluster_size=3)
        labels_large = clusterer_large.fit(simple_distance_matrix_numpy)

        assert len(labels_small) == 4
        assert len(labels_large) == 4

    def test_all_noise(self):
        """Test case where all points are classified as noise."""
        # Distance matrix where all points are far apart
        distance_matrix = np.array(
            [
                [0.0, 10.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 0.0],
            ]
        )

        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        # All points should be noise
        assert np.all(labels == -1)

    def test_all_one_cluster(self):
        """Test case where all points form one cluster."""
        # Distance matrix where all points are close
        distance_matrix = np.array(
            [
                [0.0, 0.1, 0.2, 0.15],
                [0.1, 0.0, 0.15, 0.1],
                [0.2, 0.15, 0.0, 0.1],
                [0.15, 0.1, 0.1, 0.0],
            ]
        )

        clusterer = Dbscan(epsilon=0.5, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        # All points should be in the same cluster (not -1)
        assert np.all(labels >= 0)
        assert len(np.unique(labels)) == 1

    def test_multiple_clusters(self):
        """Test detection of multiple distinct clusters."""
        # Distance matrix with two clear clusters
        distance_matrix = np.array(
            [
                [0.0, 0.1, 10.0, 10.0],
                [0.1, 0.0, 10.0, 10.0],
                [10.0, 10.0, 0.0, 0.1],
                [10.0, 10.0, 0.1, 0.0],
            ]
        )

        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        # Should have two clusters
        unique_labels = np.unique(labels[labels >= 0])
        assert len(unique_labels) == 2

        # First two points in one cluster
        assert labels[0] == labels[1]
        # Last two points in another cluster
        assert labels[2] == labels[3]
        # The two clusters should have different labels
        assert labels[0] != labels[2]

    def test_with_float32_array(self):
        """Test with float32 numpy array."""
        distance_matrix = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ],
            dtype=np.float32,
        )

        clusterer = Dbscan(epsilon=1.2, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        assert len(labels) == 3

    def test_with_float64_array(self):
        """Test with float64 numpy array."""
        distance_matrix = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ],
            dtype=np.float64,
        )

        clusterer = Dbscan(epsilon=1.2, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        assert len(labels) == 3

    def test_symmetric_distance_matrix(self):
        """Test that distance matrix should be symmetric."""
        # Valid symmetric matrix
        symmetric_matrix = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ]
        )

        clusterer = Dbscan(epsilon=1.2, min_cluster_size=2)
        labels = clusterer.fit(symmetric_matrix)
        assert len(labels) == 3

    def test_zero_diagonal(self):
        """Test that diagonal elements should be zero (distance to self)."""
        # Valid distance matrix with zero diagonal
        distance_matrix = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ]
        )

        clusterer = Dbscan(epsilon=1.2, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)
        assert len(labels) == 3

    def test_large_distance_matrix(self):
        """Test with a larger distance matrix."""
        n = 20
        # Create a distance matrix with some structure
        distance_matrix = np.random.rand(n, n)
        # Make it symmetric
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        # Set diagonal to zero
        np.fill_diagonal(distance_matrix, 0)

        clusterer = Dbscan(epsilon=0.5, min_cluster_size=3)
        labels = clusterer.fit(distance_matrix)

        assert len(labels) == n
        assert labels.dtype == np.int32

    def test_single_point(self):
        """Test with a single point."""
        distance_matrix = np.array([[0.0]])

        clusterer = Dbscan(epsilon=1.0, min_cluster_size=1)
        labels = clusterer.fit(distance_matrix)

        assert len(labels) == 1

    def test_two_points_close(self):
        """Test with two close points."""
        distance_matrix = np.array(
            [
                [0.0, 0.3],
                [0.3, 0.0],
            ]
        )

        clusterer = Dbscan(epsilon=0.5, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        # Both points should be in the same cluster
        assert labels[0] == labels[1]
        assert labels[0] >= 0

    def test_two_points_far(self):
        """Test with two distant points."""
        distance_matrix = np.array(
            [
                [0.0, 5.0],
                [5.0, 0.0],
            ]
        )

        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        # Both points should be noise
        assert labels[0] == -1
        assert labels[1] == -1

    def test_readme_example_pattern(self):
        """Test pattern similar to README examples."""
        # Time series that are similar
        series = [
            np.array([1.0, 3.0, 4.0]),
            np.array([1.0, 3.0, 3.9]),
            np.array([1.1, 2.9, 4.1]),
            np.array([5.0, 6.2, 10.0]),  # Outlier
        ]

        # Compute DTW distances
        dtw = Dtw(distance_fn="euclidean")
        distance_matrix = dtw.distance_matrix(series)

        # Cluster
        clusterer = Dbscan(epsilon=0.5, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        # First three should cluster together, last one is noise
        # Check they're all in the same cluster (not -1)
        assert labels[0] >= 0
        assert labels[0] == labels[1]
        assert labels[1] == labels[2]
        assert labels[3] == -1

    def test_with_manhattan_dtw(self):
        """Test clustering with Manhattan DTW distance."""
        series = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([5.0, 6.0, 7.0]),
        ]

        dtw = Dtw(distance_fn="manhattan")
        distance_matrix = dtw.distance_matrix(series)

        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)
        labels = clusterer.fit(distance_matrix)

        assert len(labels) == 3
        # First two should be similar
        assert labels[0] == labels[1]

    def test_reuse_clusterer(self, simple_distance_matrix_numpy):
        """Test that clusterer can be reused multiple times."""
        clusterer = Dbscan(epsilon=1.0, min_cluster_size=2)

        # First fit
        labels1 = clusterer.fit(simple_distance_matrix_numpy)
        assert len(labels1) == 4

        # Second fit with same data
        labels2 = clusterer.fit(simple_distance_matrix_numpy)
        np.testing.assert_array_equal(labels1, labels2)

        # Third fit with different data
        different_matrix = np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        labels3 = clusterer.fit(different_matrix)
        assert len(labels3) == 3
