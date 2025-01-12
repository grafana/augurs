# Time Series Clustering
Time series clustering is a technique used to group similar time series together. This can be useful for finding patterns in data, detecting anomalies, or reducing the dimensionality of large datasets.

`augurs` provides several clustering algorithms, including DBSCAN (Density-Based Spatial Clustering of Applications with Noise). DBSCAN is particularly well-suited for time series data as it:

- Doesn't require specifying the number of clusters upfront
- Can find arbitrarily shaped clusters
- Can identify noise points that don't belong to any cluster
- Works well with Dynamic Time Warping (DTW) distance measures

## Basic Example

Let's start with a simple example using DBSCAN clustering:

```rust
# extern crate augurs;
use augurs::{
    clustering::{DbscanCluster, DbscanClusterer},
    dtw::Dtw,
};

// Sample time series data
const SERIES: &[&[f64]] = &[
    &[0.0, 1.0, 2.0, 3.0, 4.0],
    &[0.1, 1.1, 2.1, 3.1, 4.1],
    &[5.0, 6.0, 7.0, 8.0, 9.0],
    &[5.1, 6.1, 7.1, 8.1, 9.1],
    &[10.0, 11.0, 12.0, 13.0, 14.0],
];

fn main() {
    // Compute distance matrix using DTW
    let distance_matrix = Dtw::euclidean()
        .with_window(2)
        .with_lower_bound(4.0)
        .with_upper_bound(10.0)
        .with_max_distance(10.0)
        .distance_matrix(SERIES);

    // Set DBSCAN parameters
    let epsilon = 0.5;
    let min_cluster_size = 2;

    // Perform clustering
    let clusters = DbscanClusterer::new(epsilon, min_cluster_size)
        .fit(&distance_matrix);

    // Clusters are labeled: -1 for noise, 0+ for cluster membership
    assert_eq!(
        clusters,
        vec![
            DbscanCluster::Cluster(1.try_into().unwrap()),
            DbscanCluster::Cluster(1.try_into().unwrap()),
            DbscanCluster::Cluster(2.try_into().unwrap()),
            DbscanCluster::Cluster(2.try_into().unwrap()),
            DbscanCluster::Noise,
        ]
    );
}
```

## Understanding Parameters

### DTW Parameters

- `window`: Size of the Sakoe-Chiba band for constraining DTW computation
- `lower_bound`: Minimum distance to consider
- `upper_bound`: Maximum distance to consider
- `max_distance`: Early termination threshold

### DBSCAN Parameters

- `epsilon`: Maximum distance between two points for one to be considered in the neighborhood of the other
- `min_cluster_size`: Minimum number of points required to form a dense region

## Best Practices

1. **Distance Measure Selection**
   - Use DTW for time series that might be shifted or warped
   - Consider the computational cost of DTW for large datasets
   - Experiment with different window sizes to balance accuracy and performance

2. **Parameter Tuning**
   - Start with a relatively large `epsilon` and reduce it if clusters are too large
   - Set `min_cluster_size` based on your domain knowledge
   - Use the DTW window parameter to prevent pathological alignments

3. **Performance Optimization**
   - Enable parallel processing for large datasets
   - Use DTW bounds to speed up distance calculations
   - Consider downsampling very long time series

## Example: Clustering with Multiple Distance Measures

```rust
# extern crate augurs;
use augurs::{
    clustering::DbscanClusterer,
    dtw::{Dtw, Distance}
};

fn compare_distance_measures(series: &[&[f64]]) {
    // Euclidean DTW
    let euclidean_matrix = Dtw::euclidean()
        .distance_matrix(series);
    let euclidean_clusters = DbscanClusterer::new(0.5, 2)
        .fit(&euclidean_matrix);

    // Manhattan DTW
    let manhattan_matrix = Dtw::manhattan()
        .distance_matrix(series);
    let manhattan_clusters = DbscanClusterer::new(0.5, 2)
        .fit(&manhattan_matrix);

    // Compare results
    println!("Euclidean clusters: {:?}", euclidean_clusters);
    println!("Manhattan clusters: {:?}", manhattan_clusters);
}
```

## Next Steps

- Learn about [outlier detection](../how-to/outliers.md) using clustering
- Explore [seasonality analysis](../how-to/seasonality.md) for clustered time series
- Understand [feature extraction](../how-to/features.md) for time series
