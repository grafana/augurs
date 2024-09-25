//! This example demonstrates how to use DBSCAN clustering to identify clusters in a time series.
//!
//! The example uses Dynamic Time Warping using the Euclidean distance to compute
//! a distance matrix between the time series. The distance matrix is then
//! passed to the DBSCAN clustering algorithm.
//!
//! The resulting clusters are assigned a label of -1 for noise, 0 for the first cluster, and 1 for
//! the second cluster.

use augurs::{clustering::DbscanClusterer, dtw::Dtw};

// This is a very trivial example dataset containing 5 time series which
// form two obvious clusters, plus a noise cluster.
const SERIES: &[&[f64]] = &[
    &[0.0, 1.0, 2.0, 3.0, 4.0],
    &[0.1, 1.1, 2.1, 3.1, 4.1],
    &[5.0, 6.0, 7.0, 8.0, 9.0],
    &[5.1, 6.1, 7.1, 8.1, 9.1],
    &[10.0, 11.0, 12.0, 13.0, 14.0],
];

fn main() {
    // Use dynamic time warping to compute a distance matrix between the two time series.
    // We'll use the Euclidean distance.
    let distance_matrix = Dtw::euclidean()
        // DTW has a few options that can be specified to speed up calculations
        // for large numbers of time series. See the documentation for more
        // information.
        // Optionally specify the size of the Sakoe-Chiba window.
        .with_window(2)
        // Optionally specify a lower or upper bound for the distances. This can
        // speed up calculations by ignoring distances below the lower bound.
        .with_lower_bound(4.0)
        // Optionally specify an upper bound for the distances. This can speed up
        // calculations by ignoring distances above the upper bound.
        .with_upper_bound(10.0)
        // Optionally specify the maximum distance between two time series.
        // This can speed up calculations by terminating early if the maximum
        // distance is exceeded.
        .with_max_distance(10.0)
        // Compute the distance matrix between all pairs of time series.
        .distance_matrix(SERIES);

    // Epsilon is the maximum distance between two time series for one to be considered as in the
    // neighborhood of the other.
    let epsilon = 0.5;

    // Minimum number of points in a neighborhood for a point to be considered as a core point.
    let min_cluster_size = 2;

    // Run DBSCAN clustering on the distance matrix.
    let clusters: Vec<isize> =
        DbscanClusterer::new(epsilon, min_cluster_size).fit(&distance_matrix);
    assert_eq!(clusters, vec![0, 0, 1, 1, -1]);
}
