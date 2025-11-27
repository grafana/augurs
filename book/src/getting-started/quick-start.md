# Quick Start Guide

This guide will help you get started with `augurs`, a time series toolkit available for Rust, JavaScript, and Python.

## Installation

<!-- langtabs-start -->
```rust
// Add to your Cargo.toml:
[dependencies]
augurs = { version = "0.6.0", features = ["forecaster", "ets", "mstl", "outlier"] }
```

```javascript
// Install via npm:
npm install @bsull/augurs
```

```python
# Install via pip:
pip install augurs
```
<!-- langtabs-end -->

## Basic Forecasting

Let's start with a simple example using the MSTL (Multiple Seasonal-Trend decomposition using LOESS) model
and a naive trend forecaster:

<!-- langtabs-start -->
```rust
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

```javascript
import { MSTL } from '@bsull/augurs/mstl';

// Sample time series data
const data = [1.0, 1.2, 1.4, 1.5, 1.4, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8, 1.9, 2.0];

// Create an MSTL model with weekly seasonality (period = 7)
const mstl = MSTL.ets([7]);

// Fit the model
mstl.fit(data);

// Generate forecasts with 95% prediction intervals
const forecast = mstl.predict(10, 0.95);

console.log("Forecast values:", forecast.point);
console.log("Lower bounds:", forecast.intervals.lower);
console.log("Upper bounds:", forecast.intervals.upper);
```

```python
import augurs as aug
import numpy as np

# Sample time series data
data = np.array([1.0, 1.2, 1.4, 1.5, 1.4, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8, 1.9, 2.0])

# Create an MSTL model with weekly seasonality (period = 7)
mstl = aug.MSTL.ets([7])

# Fit the model
mstl.fit(data)

# Generate forecasts with 95% prediction intervals
forecast = mstl.predict(10, level=0.95)

print("Forecast values:", forecast.point())
print("Lower bounds:", forecast.lower())
print("Upper bounds:", forecast.upper())
```
<!-- langtabs-end -->

## Advanced Forecasting with Transforms

For more complex scenarios, you can use the `Forecaster` API which supports data transformations:

<!-- langtabs-start -->
```rust
use augurs::{
    ets::AutoETS,
    forecaster::{
        transforms::{LinearInterpolator, Log, MinMaxScaler},
        Forecaster, Transformer,
    },
    mstl::MSTLModel,
};

fn main() {
    let data = &[1.0, 1.2, 1.4, 1.5, f64::NAN, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8];

    // Set up model and transformers
    let ets = AutoETS::non_seasonal().into_trend_model();
    let mstl = MSTLModel::new(vec![2], ets);

    let transformers = vec![
        LinearInterpolator::new().boxed(),
        MinMaxScaler::new().boxed(),
        Log::new().boxed(),
    ];

    // Create and fit forecaster
    let mut forecaster = Forecaster::new(mstl).with_transformers(transformers);
    forecaster.fit(data).expect("model should fit");

    // Generate forecasts
    let forecast = forecaster
        .predict(5, 0.95)
        .expect("forecasting should work");
}
```

```javascript
import { AutoETS } from '@bsull/augurs/ets';

const data = [1.0, 1.2, 1.4, 1.5, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8];

// Create an AutoETS model
const ets = new AutoETS(3, 'ZZN');

// Fit the model
ets.fit(data);

// Generate forecasts
const forecast = ets.predict(5, 0.95);

console.log("Forecast values:", forecast.point);
```

```python
import augurs as aug
import numpy as np

data = np.array([1.0, 1.2, 1.4, 1.5, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8])

# Create an AutoETS model
# season_length=3, spec='ZZN' (auto error, auto trend, no seasonality)
ets = aug.AutoETS(3, 'ZZN')

# Fit the model
ets.fit(data)

# Generate forecasts with 95% prediction intervals
forecast = ets.predict(5, level=0.95)

print("Forecast values:", forecast.point())
print("Lower bounds:", forecast.lower())
print("Upper bounds:", forecast.upper())
```
<!-- langtabs-end -->

> **Note**: The `Forecaster` API with data transformations shown in the Rust example is currently only available in Rust. JavaScript and Python examples show basic AutoETS usage instead.

## Outlier Detection

`augurs` provides multiple algorithms for outlier detection. Here's an example using the MAD (Median Absolute Deviation) detector:

<!-- langtabs-start -->
```rust
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

```javascript
import { OutlierDetector } from '@bsull/augurs/outlier';

const series = [
    [1.0, 2.0, 1.5, 2.3],
    [1.9, 2.2, 1.2, 2.4],
    [1.5, 2.1, 6.4, 8.5], // This series contains outliers
];

// Create MAD detector with sensitivity 0.5
const detector = OutlierDetector.mad({ sensitivity: 0.5 });

// Detect outliers
const outliers = detector.detect(series);

console.log("Outlying series indices:", outliers.outlyingSeries);
```

```python
import augurs as aug
import numpy as np

series = [
    np.array([1.0, 2.0, 1.5, 2.3]),
    np.array([1.9, 2.2, 1.2, 2.4]),
    np.array([1.5, 2.1, 6.4, 8.5]),  # This series contains outliers
]

# Note: Outlier detection may not be available in Python bindings yet
# Check the augurs.pyi file for available methods
print("Outlier detection examples coming soon for Python!")
```
<!-- langtabs-end -->

## Time Series Clustering

You can use DBSCAN clustering with Dynamic Time Warping (DTW) distance:

<!-- langtabs-start -->
```rust
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

```javascript
import { Dtw } from '@bsull/augurs/dtw';
import { DbscanClusterer } from '@bsull/augurs/clustering';

const series = [
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [0.1, 1.1, 2.1, 3.1, 4.1],
    [5.0, 6.0, 7.0, 8.0, 9.0],
];

// Compute distance matrix using DTW
const dtw = new Dtw('euclidean', { window: 2 });
const distanceMatrix = dtw.distanceMatrix(series);

// Perform clustering
const clusterer = new DbscanClusterer({ epsilon: 0.5, minClusterSize: 2 });
const clusters = clusterer.fit(distanceMatrix);

console.log("Cluster assignments:", clusters);
```

```python
import augurs as aug
import numpy as np

series = [
    np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    np.array([0.1, 1.1, 2.1, 3.1, 4.1]),
    np.array([5.0, 6.0, 7.0, 8.0, 9.0]),
]

# Compute distance matrix using DTW
dtw = aug.Dtw(window=2, distance_fn='euclidean')
distance_matrix = dtw.distance_matrix(series)

print("DTW distance matrix computed successfully!")
# Note: DistanceMatrix is a custom type, not a numpy array

# Note: Clustering may not be available in Python bindings yet
```
<!-- langtabs-end -->

## Next Steps

- Learn more about [forecasting methods](../how-to/forecasting.md)
- Explore [outlier detection algorithms](../how-to/outliers.md)
- Understand [seasonality analysis](../how-to/seasonality.md)
- Check out the [complete API documentation](../api/index.md)