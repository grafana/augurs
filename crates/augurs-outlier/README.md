# Outlier detection.

This crate provides implementations of time series _outlier detection_, the problem of determining whether one time series behaves differently to others in a group. (This is different to _anomaly detection_, which aims to determine if one or more samples appears to be different within a time series).

Two implementations are provided:

- Median Absolute Difference (MAD)
- DBSCAN
