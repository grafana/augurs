# Automated Outlier Detection

This tutorial demonstrates how to use `augurs` to automatically detect outliers in time series data. We'll explore both the MAD (Median Absolute Deviation) and DBSCAN approaches to outlier detection.

## MAD-based Outlier Detection

The MAD detector is ideal for identifying time series that deviate significantly from the typical behavior pattern:

```rust
# extern crate augurs;
use augurs::outlier::{MADDetector, OutlierDetector};

fn main() {
    // Example time series data
    let series: &[&[f64]] = &[
        &[1.0, 2.0, 1.5, 2.3],
        &[1.9, 2.2, 1.2, 2.4],
        &[1.5, 2.1, 6.4, 8.5], // This series contains outliers
    ];

    // Create detector with 50% sensitivity
    let detector = MADDetector::with_sensitivity(0.5)
        .expect("sensitivity is between 0.0 and 1.0");

    // Process and detect outliers
    let processed = detector.preprocess(series).expect("input data is valid");
    let outliers = detector.detect(&processed).expect("detection succeeds");

    println!("Outlying series indices: {:?}", outliers.outlying_series);
    println!("Series scores: {:?}", outliers.series_results);
}
```

## DBSCAN-based Outlier Detection

DBSCAN is particularly effective when your time series have seasonal patterns:

```rust
# extern crate augurs;
use augurs::outlier::{DbscanDetector, OutlierDetector};

fn main() {
    // Example time series data
    let series: &[&[f64]] = &[
        &[1.0, 2.0, 1.5, 2.3],
        &[1.9, 2.2, 1.2, 2.4],
        &[1.5, 2.1, 6.4, 8.5], // This series behaves differently
    ];

    // Create and configure detector
    let mut detector = DbscanDetector::with_sensitivity(0.5)
        .expect("sensitivity is between 0.0 and 1.0");

    // Enable parallel processing (requires 'parallel' feature)
    detector = detector.parallelize(true);

    // Process and detect outliers
    let processed = detector.preprocess(series).expect("input data is valid");
    let outliers = detector.detect(&processed).expect("detection succeeds");

    println!("Outlying series indices: {:?}", outliers.outlying_series);
    println!("Series scores: {:?}", outliers.series_results);
}
```

## Handling Results

The outlier detection results provide several useful pieces of information:

```rust
# extern crate augurs;
use augurs::outlier::{MADDetector, OutlierDetector};

fn main() {
    let series: &[&[f64]] = &[
        &[1.0, 2.0, 1.5, 2.3],
        &[1.9, 2.2, 1.2, 2.4],
        &[1.5, 2.1, 6.4, 8.5],
    ];

    let detector = MADDetector::with_sensitivity(0.5)
        .expect("sensitivity is between 0.0 and 1.0");

    let processed = detector.preprocess(series).expect("input data is valid");
    let outliers = detector.detect(&processed).expect("detection succeeds");

    // Get indices of outlying series
    for &idx in &outliers.outlying_series {
        println!("Series {} is an outlier", idx);
    }

    // Examine detailed results for each series
    for (idx, result) in outliers.series_results.iter().enumerate() {
        println!("Series {}: outlier = {}", idx, result.is_outlier);
        println!("Scores: {:?}", result.scores);
    }
}
```

## Best Practices

1. **Choosing a Detector**
   - Use MAD when you expect series to move within a stable band
   - Use DBSCAN when series may have seasonality or complex patterns

2. **Sensitivity Tuning**
   - Start with 0.5 sensitivity and adjust based on results
   - Lower values (closer to 0.0) are more sensitive
   - Higher values (closer to 1.0) are more selective

3. **Performance Optimization**
   - Enable parallelization for large datasets
   - Consider preprocessing data to remove noise
   - Handle missing values before detection

## Example: Real-time Monitoring

Here's an example of using outlier detection in a monitoring context:

```rust
# extern crate augurs;
use augurs::outlier::{MADDetector, OutlierDetector};

fn monitor_time_series(historical_data: &[&[f64]], new_data: &[f64]) -> bool {
    // Create detector from historical data
    let detector = MADDetector::with_sensitivity(0.5)
        .expect("sensitivity is between 0.0 and 1.0");

    // Combine historical and new data
    let mut all_series: Vec<&[f64]> = historical_data.to_vec();
    all_series.push(new_data);

    // Check for outliers
    let processed = detector.preprocess(&all_series)
        .expect("input data is valid");
    let outliers = detector.detect(&processed)
        .expect("detection succeeds");

    // Check if new series (last index) is an outlier
    outliers.outlying_series.contains(&(all_series.len() - 1))
}
```

This structure provides a comprehensive guide to outlier detection while maintaining a practical focus on implementation details.
