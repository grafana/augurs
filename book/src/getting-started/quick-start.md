# Quick Start Guide

This guide will help you get started with `augurs`, a Rust library for time series analysis and forecasting.

## Installation

Add `augurs` to your `Cargo.toml`:

```toml
[dependencies]
augurs = { version = "0.6.0", features = ["forecaster", "ets", "mstl", "outlier"] }
```

## Basic Forecasting

Let's start with a simple example using the MSTL (Multiple Seasonal-Trend decomposition using LOESS) model
and a naive trend forecaster:

```rust
# extern crate augurs;
use augurs::{mstl::MSTLModel, prelude::*};

fn main() {
    // Sample time series data
    let data = &[1.0, 1.2, 1.4, 1.5, 1.4, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8, 1.9, 2.0];

    // Create an MSTL model with weekly seasonality (period = 7)
    let mstl = MSTLModel::naive(vec![7]);

    // Fit the model
    let fit = mstl.fit(data).expect("model should fit");

    // Generate forecasts with 95% prediction intervals
    let forecast = fit
        .predict(10, 0.95)
        .expect("forecasting should work");

    println!("Forecast values: {:?}", forecast.point);
    println!("Lower bounds: {:?}", forecast.intervals.as_ref().unwrap().lower);
    println!("Upper bounds: {:?}", forecast.intervals.as_ref().unwrap().upper);
}
```

## Advanced Forecasting with Transforms

For more complex scenarios, you can use the `Forecaster` API which supports data transformations:

```rust
# extern crate augurs;
use augurs::{
    ets::AutoETS,
    forecaster::{
        transforms::{LinearInterpolator, Log, MinMaxScaler},
        Forecaster, Transform,
    },
    mstl::MSTLModel,
};

fn main() {
    let data = &[1.0, 1.2, 1.4, 1.5, f64::NAN, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8];

    // Set up model and transforms
    let ets = AutoETS::non_seasonal().into_trend_model();
    let mstl = MSTLModel::new(vec![2], ets);

    let transforms = vec![
        LinearInterpolator::new().boxed(),
        MinMaxScaler::new().boxed(),
        Log::new().boxed(),
    ];

    // Create and fit forecaster
    let mut forecaster = Forecaster::new(mstl).with_transforms(transforms);
    forecaster.fit(data).expect("model should fit");

    // Generate forecasts
    let forecast = forecaster
        .predict(5, 0.95)
        .expect("forecasting should work");
}
```

## Outlier Detection

`augurs` provides multiple algorithms for outlier detection. Here's an example using the MAD (Median Absolute Deviation) detector:

```rust
# extern crate augurs;
use augurs::outlier::{MADDetector, OutlierDetector};

fn main() {
    let series: &[&[f64]] = &[
        &[1.0, 2.0, 1.5, 2.3],
        &[1.9, 2.2, 1.2, 2.4],
        &[1.5, 2.1, 6.4, 8.5], // This series contains outliers
    ];

    // Create and configure detector
    let detector = MADDetector::with_sensitivity(0.5)
        .expect("sensitivity is between 0.0 and 1.0");

    // Detect outliers
    let processed = detector.preprocess(series).expect("input data is valid");
    let outliers = detector.detect(&processed).expect("detection succeeds");

    println!("Outlying series indices: {:?}", outliers.outlying_series);
}
```

## Time Series Clustering

You can use DBSCAN clustering with Dynamic Time Warping (DTW) distance:

```rust
# extern crate augurs;
use augurs::{clustering::DbscanClusterer, dtw::Dtw};

fn main() {
    let series: &[&[f64]] = &[
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &[0.1, 1.1, 2.1, 3.1, 4.1],
        &[5.0, 6.0, 7.0, 8.0, 9.0],
    ];

    // Compute distance matrix using DTW
    let distance_matrix = Dtw::euclidean()
        .with_window(2)
        .distance_matrix(series);

    // Perform clustering
    let clusters = DbscanClusterer::new(0.5, 2)
        .fit(&distance_matrix);

    println!("Cluster assignments: {:?}", clusters);
}
```

## Next Steps

- Learn more about [forecasting methods](../how-to/forecasting.md)
- Explore [outlier detection algorithms](../how-to/outliers.md)
- Understand [seasonality analysis](../how-to/seasonality.md)
- Check out the [complete API documentation](../api/index.md)
