# Time series clustering algorithms

This crate contains algorithms for clustering time series.

So far only DBSCAN is implemented, and the distance matrix must be passed directly.
A crate such as [`augurs-dtw`] must be used to calculate the distance matrix for now.

## Usage

```rust
use augurs_clustering::{Dbscan, DistanceMatrix};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let distance_matrix = DistanceMatrix::try_from_square(
    vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 0.0, 3.0, 3.0],
        vec![2.0, 3.0, 0.0, 4.0],
        vec![3.0, 3.0, 4.0, 0.0],
    ],
)?;
let clusters = Dbscan::new(0.5, 2).fit(&distance_matrix);
assert_eq!(clusters, vec![-1, -1, -1, -1]);
# Ok(())
# }
```

## Credits

This implementation based heavily on to the implementation in [`linfa-clustering`] and [`scikit-learn`].
The main difference between these is that we operate directly on the distance matrix rather than calculating
it as part of the clustering algorithm.

[`augurs-dtw`]: https://crates.io/crates/augurs-dtw
[`linfa-clustering`]: https://crates.io/crates/linfa-clustering
[`scikit-learn`]: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

## License

Dual-licensed to be compatible with the Rust project.
Licensed under the Apache License, Version 2.0 `<http://www.apache.org/licenses/LICENSE-2.0>` or the MIT license `<http://opensource.org/licenses/MIT>`, at your option.
