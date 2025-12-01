# Automated Outlier Detection

This tutorial demonstrates how to use `augurs` to automatically detect outliers in time series data. We'll explore both the MAD (Median Absolute Deviation) and DBSCAN approaches to outlier detection.

## MAD-based Outlier Detection

The MAD detector is ideal for identifying time series that deviate significantly from the typical behavior pattern:

<!-- langtabs-start -->
```rust
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

```javascript
import { OutlierDetector } from '@bsull/augurs/outlier';

// Example time series data
const series = [
    [1.0, 2.0, 1.5, 2.3],
    [1.9, 2.2, 1.2, 2.4],
    [1.5, 2.1, 6.4, 8.5], // This series contains outliers
];

// Create MAD detector with 50% sensitivity
const detector = OutlierDetector.mad({ sensitivity: 0.5 });

// Detect outliers
const outliers = detector.detect(series);

console.log("Outlying series indices:", outliers.outlyingSeries);
console.log("Series scores:", outliers.seriesResults);
```


<!-- langtabs-end -->

## DBSCAN-based Outlier Detection

DBSCAN is particularly effective when your time series have seasonal patterns:

<!-- langtabs-start -->
```rust
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

```javascript
import { OutlierDetector } from '@bsull/augurs/outlier';

// Example time series data
const series = [
    [1.0, 2.0, 1.5, 2.3],
    [1.9, 2.2, 1.2, 2.4],
    [1.5, 2.1, 6.4, 8.5], // This series behaves differently
];

// Create DBSCAN detector with 50% sensitivity
const detector = OutlierDetector.dbscan({ sensitivity: 0.5 });

// Detect outliers
const outliers = detector.detect(series);

console.log("Outlying series indices:", outliers.outlyingSeries);
console.log("Series scores:", outliers.seriesResults);
```


<!-- langtabs-end -->

## Handling Results

The outlier detection results provide several useful pieces of information:

<!-- langtabs-start -->
```rust
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

```javascript
import { OutlierDetector } from '@bsull/augurs/outlier';

const series = [
    [1.0, 2.0, 1.5, 2.3],
    [1.9, 2.2, 1.2, 2.4],
    [1.5, 2.1, 6.4, 8.5],
];

const detector = OutlierDetector.mad({ sensitivity: 0.5 });
const outliers = detector.detect(series);

// Get indices of outlying series
for (const idx of outliers.outlyingSeries) {
    console.log(`Series ${idx} is an outlier`);
}

// Examine detailed results for each series
outliers.seriesResults.forEach((result, idx) => {
    console.log(`Series ${idx}: outlier = ${result.isOutlier}`);
    console.log(`Scores:`, result.scores);
});
```


<!-- langtabs-end -->

## Best Practices

1. **Choosing a Detector**
   - Use MAD when you expect series to move within a stable band
   - Use DBSCAN when series may have seasonality or complex patterns

2. **Sensitivity Tuning**
   - Start with 0.5 sensitivity and adjust based on results
   - Lower values (closer to 0.0) are more sensitive
   - Higher values (closer to 1.0) are more selective

3. **Performance Optimization**
   - Enable parallelization for large datasets (Rust)
   - Consider preprocessing data to remove noise
   - Handle missing values before detection

4. **Language-Specific Considerations**
   - **Rust**: Full feature set with compile-time safety
   - **JavaScript**: WASM-based, good for browser and Node.js
   - **Python**: Outlier detection not yet available in Python bindings

## Example: Real-time Monitoring

Here's an example of using outlier detection in a monitoring context:

<!-- langtabs-start -->
```rust
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

fn main() {
    let historical = vec![
        &[1.0, 2.0, 1.5][..],
        &[1.9, 2.2, 1.2][..],
    ];
    let new = &[1.5, 2.1, 6.4];
    
    if monitor_time_series(&historical, new) {
        println!("Alert: Anomaly detected!");
    }
}
```

```javascript
import { OutlierDetector } from '@bsull/augurs/outlier';

function monitorTimeSeries(historicalData, newData) {
    // Create detector
    const detector = OutlierDetector.mad({ sensitivity: 0.5 });

    // Combine historical and new data
    const allSeries = [...historicalData, newData];

    // Check for outliers
    const outliers = detector.detect(allSeries);

    // Check if new series (last index) is an outlier
    return outliers.outlyingSeries.includes(allSeries.length - 1);
}

// Example usage
const historical = [
    [1.0, 2.0, 1.5],
    [1.9, 2.2, 1.2],
];
const newData = [1.5, 2.1, 6.4];

if (monitorTimeSeries(historical, newData)) {
    console.log("Alert: Anomaly detected!");
}
```


<!-- langtabs-end -->

## Next Steps

- Explore [clustering](./clustering.md) to group similar time series
- Learn about [forecasting](./forecasting-with-prophet.md) for predictive analytics
- Check the [API documentation](../api/index.md) for advanced features

This comprehensive guide provides practical examples for implementing outlier detection across different programming languages, with clear guidance on best practices and real-world applications.
