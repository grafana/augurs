# Outlier detection.

This crate provides implementations of time series _outlier detection_, the problem of determining whether one time series behaves differently to others in a group. (This is different to _anomaly detection_, which aims to determine if one or more samples appears to be different within a time series).

Two implementations are planned:

- DBSCAN: implemented
- Median Absolute Difference (MAD): not yet implemented (see [GitHub issue](https://github.com/grafana/augurs/issues/82))

# Example

```rust
use augurs_outlier::{OutlierDetector, DBSCANDetector};

// Each slice inside `data` is a time series.
// The third one behaves differently at indexes 2 and 3.
let data: &[&[f64]] = &[
    &[1.0, 2.0, 1.5, 2.3],
    &[1.9, 2.2, 1.2, 2.4],
    &[1.5, 2.1, 6.4, 8.5],
];
let detector = DBSCANDetector::with_sensitivity(0.5)
    .expect("sensitivity is between 0.0 and 1.0");
let processed = detector.preprocess(data);
let outliers = detector.detect(&processed);

assert_eq!(outliers.outlying_series.len(), 1);
assert!(outliers.outlying_series.contains(&2));
assert!(outliers.series_results[2].is_outlier);
assert_eq!(outliers.series_results[2].scores, vec![0.0, 0.0, 1.0, 1.0]);
```
