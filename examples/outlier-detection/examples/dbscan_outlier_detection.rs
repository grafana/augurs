//! This example demonstrates how to use the DBSCAN outlier detection algorithm.
//!
//! Outlier detection is a technique used to identify time series which behave
//! differently from the rest of the series in the data.
//!
//! DBSCAN is a good choice of algorithm when the series in the data may have
//! seasonality and when they are expected to 'move together' over time.

use augurs::outlier::{DbscanDetector, OutlierDetector};

// Each inner slice below is a time series.
// The third one behaves differently at indexes 2 and 3.
const SERIES: &[&[f64]] = &[
    &[1.0, 2.0, 1.5, 2.3],
    &[1.9, 2.2, 1.2, 2.4],
    &[1.5, 2.1, 6.4, 8.5],
];

fn main() {
    // Set the sensitivity of the detector.
    // Sensitivity is a number between 0.0 and 1.0 which will be converted to a
    // sensible value for the `epsilon` parameter of the DBSCAN algorithm
    // depending on the span of the data.
    let sensitivity = 0.5;

    // Create a new DBSCAN detector with the given sensitivity.
    // This function will return an error if the sensitivity is not between 0.0 and 1.0.
    let mut detector =
        DbscanDetector::with_sensitivity(sensitivity).expect("sensitivity is between 0.0 and 1.0");

    // Optionally, turn on parallel processing.
    // This requires the `parallel` feature to be enabled, otherwise it will be ignored.
    detector = detector.parallelize(true);

    // Preprocess the data using the detector.
    // This function will return an error if the input data is invalid.
    let processed = DbscanDetector::preprocess(SERIES).expect("input data is valid");

    // Note: we could also have created the preprocessed data from 'column major' data,
    // using `DbscanData::from_column_major`.
    // let preprocessed = augurs::outlier::DbscanData::from_column_major(TRANSPOSED_SERIES);

    // Detect outliers in the preprocessed data.
    // This function will return an error if the detection fails; this may be impossible
    // for certain detector implementations.
    let outliers = detector.detect(&processed).expect("detection succeeds");

    // We have one outlying series, which is the third one.
    assert_eq!(outliers.outlying_series.len(), 1);
    assert!(outliers.outlying_series.contains(&2));
    // `outliers.series_results` contains the results of the outlier detection for each series.
    assert_eq!(outliers.series_results.len(), 3);
    assert!(outliers.series_results[2].is_outlier);
    // For a DBSCAN detector, the scores are either 0.0 (for non-outlying
    // time points) or 1.0 (for outlying time points).
    assert_eq!(outliers.series_results[2].scores, vec![0.0, 0.0, 1.0, 1.0]);
}
