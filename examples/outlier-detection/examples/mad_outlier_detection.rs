//! This example demonstrates how to use the MAD outlier detection algorithm.
//!
//! Outlier detection is a technique used to identify time series which behave
//! differently from the rest of the series in the data.
//!
//! MAD is a good choice of algorithm when the you expect the series in your data
//! to move within a stable band of normal behaviour.

use augurs::outlier::{MADDetector, OutlierDetector};

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
    // sensible value for the `threshold` parameter of the MAD algorithm
    // depending on the span of the data.
    let sensitivity = 0.5;

    // Create a new MAD detector with the given sensitivity.
    // This function will return an error if the sensitivity is not between 0.0 and 1.0.
    let detector =
        MADDetector::with_sensitivity(sensitivity).expect("sensitivity is between 0.0 and 1.0");

    // Preprocess the data using the detector.
    // This function will return an error if the input data is invalid.
    let processed = MADDetector::preprocess(SERIES).expect("input data is valid");

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
    // For a MAD detector the scores are the median absolute deviations themselves.
    assert_eq!(
        outliers.series_results[2].scores,
        vec![
            0.6835259767082061,
            0.057793242408848366,
            5.028012089569781,
            7.4553282707414
        ]
    );
}
