# Time series clustering algorithms

Time series clustering algorithms.

So far, only DBSCAN is implemented, and the distance matrix must be passed directly.
A crate such as [`augurs-dtw`] must be used to calculate the distance matrix for now.

## Usage

```rust
use augurs_clustering::{DbscanClusterer, DistanceMatrix};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Start with a distance matrix.
// This can be calculated using e.g. dynamic time warping
// using the `augurs-dtw` crate.
let distance_matrix = DistanceMatrix::try_from_square(
    vec![
        vec![0.0, 0.1, 0.2, 2.0, 1.9],
        vec![0.1, 0.0, 0.15, 2.1, 2.2],
        vec![0.2, 0.15, 0.0, 2.2, 2.3],
        vec![2.0, 2.1, 2.2, 0.0, 0.1],
        vec![1.9, 2.2, 2.3, 0.1, 0.0],
    ],
)?;

// Epsilon is the maximum distance between two series for them to be considered in the same cluster.
let epsilon = 0.3;
// The minimum number of series in a cluster before it is considered non-noise.
let min_cluster_size = 2;

// Use DBSCAN to detect clusters of series.
// Note that we don't need to specify the number of clusters in advance.
let clusters = DbscanClusterer::new(epsilon, min_cluster_size).fit(&distance_matrix);
assert_eq!(clusters, vec![0, 0, 0, 1, 1]);
# Ok(())
# }
```

## Credits

This implementation is based heavily on to the implementation in [`linfa-clustering`] and [`scikit-learn`].
The main difference between these is that we operate directly on the distance matrix rather than calculating
it as part of the clustering algorithm.

[`augurs-dtw`]: https://crates.io/crates/augurs-dtw
[`linfa-clustering`]: https://crates.io/crates/linfa-clustering
[`scikit-learn`]: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

## License

Dual-licensed to be compatible with the Rust project.
Licensed under the Apache License, Version 2.0 `<http://www.apache.org/licenses/LICENSE-2.0>` or the MIT license `<http://opensource.org/licenses/MIT>`, at your option.
