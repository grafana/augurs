# Core Concepts

`augurs` is a comprehensive time series analysis library that provides several core capabilities:

## Forecasting
Time series forecasting involves predicting future values based on historical patterns. The library supports multiple forecasting methods:
- MSTL (Multiple Seasonal-Trend decomposition using LOESS)
- ETS (Error, Trend, Seasonal) models
- Prophet (Facebook's forecasting tool)
- Custom models through the `Forecaster` trait

## Clustering
Time series clustering helps identify groups of similar time series within a dataset. Key features include:
- DBSCAN clustering with DTW (Dynamic Time Warping) distance
- Flexible distance metrics
- Parallel processing support for large datasets

## Outlier Detection
Outlier detection is the task of identifying one or more time series that deviate significantly from the norm. `augurs` includes:
- MAD (Median Absolute Deviation) detection
- DBSCAN-based outlier detection
- Customizable sensitivity parameters

## Changepoint Detection
`augurs` re-exports the `changepoint` crate for detecting changes in time series data:
- Normal distribution-based changepoint detection
- Autoregressive Gaussian process changepoint detection

## Seasonality Analysis
Understanding seasonal patterns is essential for time series analysis:
- Automatic period detection
- Multiple seasonality handling
- Seasonal decomposition

## Data Transformations
The library supports various data transformations:
- Linear interpolation for missing values
- Min-max scaling
- Logarithmic transformation
- Custom transformations through the `Transform` trait
