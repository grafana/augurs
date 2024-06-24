#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

use std::collections::HashSet;

mod dbscan;
mod mad;
#[cfg(test)]
mod testing;

pub use dbscan::DBSCANDetector;

/// The sensitivity of an outlier detection algorithm.
///
/// Sensitivity values are between 0.0 and 1.0, where 0.0 means
/// the algorithm is not sensitive to outliers and 1.0 means the
/// algorithm is very sensitive to outliers.
///
/// The exact meaning of the sensitivity value depends on the
/// implementation of the outlier detection algorithm.
/// For example, a DBSCAN based algorithm might use the sensitivity
/// to determine the maximum distance between points in the same
/// cluster (i.e. `epsilon`).
///
/// Crucially, though, sensitivity will always be a value between 0.0
/// and 1.0 to make it easier to reason about for users.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct Sensitivity(f64);

impl TryFrom<f64> for Sensitivity {
    type Error = SensitivityError;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value <= 0.0 || value >= 1.0 {
            Err(SensitivityError(value))
        } else {
            Ok(Self(value))
        }
    }
}

/// An error indicating that the sensitivity value is out of bounds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SensitivityError(f64);

impl std::fmt::Display for SensitivityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sensitivity must be between 0.0 and 1.0, got {}", self.0)
    }
}

impl std::error::Error for SensitivityError {}

/// A band indicating the min and max value considered outlying
/// at each timestamp.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Band {
    /// The minimum value considered outlying at each timestamp.
    pub min: Vec<f64>,
    /// The maximum value considered outlying at each timestamp.
    pub max: Vec<f64>,
}

impl Band {
    fn new(n_timestamps: usize) -> Self {
        Self {
            min: vec![f64::NAN; n_timestamps],
            max: vec![f64::NAN; n_timestamps],
        }
    }
}

/// The result of applying an outlier detection algorithm to a time series.
#[derive(Debug, Clone)]
pub struct OutlierResult {
    /// The indexes of the series considered outliers.
    pub outlying_series: HashSet<usize>,

    /// The results of the detection for each series.
    pub series_results: Vec<Series>,

    /// The band indicating the min and max value considered outlying
    /// at each timestamp.
    pub cluster_band: Band,
}

impl OutlierResult {
    /// Create a new `OutlierResult` from the given series results.
    pub fn new(series_results: Vec<Series>, cluster_band: Band) -> Self {
        Self {
            outlying_series: series_results
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.is_outlier.then_some(i))
                .collect(),
            series_results,
            cluster_band,
        }
    }

    /// Determine whether the series at the given index is an outlier.
    pub fn is_outlier(&self, i: usize) -> bool {
        self.outlying_series.contains(&i)
    }
}

/// A potentially outlying series.
#[derive(Debug, Clone)]
pub struct Series {
    /// Whether the series is an outlier for at least one of the samples.
    pub is_outlier: bool,
    /// The intervals of the samples that are considered outliers.
    pub outlier_intervals: OutlierIntervals,
    /// The outlier scores of the series for each sample.
    ///
    /// The higher the score, the more likely the series is an outlier.
    /// Note that some implementations may not provide continuous scores
    /// but rather a binary classification. In this case, the scores will
    /// be 0.0 for non-outliers and 1.0 for outliers.
    pub scores: Vec<f64>,
}

impl Series {
    /// Create a new non-outlying `Series` with an empty set of scores.
    pub fn empty() -> Self {
        Self {
            is_outlier: false,
            scores: Vec::new(),
            outlier_intervals: OutlierIntervals::empty(),
        }
    }

    /// Create a new non-outlying `Series` with an empty set of scores
    /// and the given capacity.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            is_outlier: false,
            scores: Vec::with_capacity(n),
            outlier_intervals: OutlierIntervals::empty(),
        }
    }
}

/// A list of outlier intervals for a single series.
///
/// We manually implement [`Serialize`] for this struct, serializing
/// just the `timestamps` field as an array.
#[derive(Debug, Clone)]
pub struct OutlierIntervals {
    /// A list of indices, where 'even' elements are the start and
    /// 'odd' elements are the end of an outlier interval.
    pub indices: Vec<usize>,

    /// Are we expecting a start or end timestamp to be pushed next?
    expecting_end: bool,
}

impl OutlierIntervals {
    // fn new(idx: usize) -> Self {
    //     // We're expecting at least two indices, so we might
    //     // as well allocate it now.
    //     let mut indices = Vec::with_capacity(2);
    //     indices.push(idx);
    //     Self {
    //         indices,
    //         expecting_end: true,
    //     }
    // }

    fn empty() -> Self {
        Self {
            indices: Vec::new(),
            expecting_end: false,
        }
    }

    fn add_start(&mut self, ts: usize) {
        debug_assert!(
            !self.expecting_end,
            "Expected end of outlier interval, got start"
        );
        self.indices.push(ts);
        self.expecting_end = true;
    }

    fn add_end(&mut self, ts: usize) {
        debug_assert!(
            self.expecting_end,
            "Expected start of outlier interval, got end"
        );
        self.indices.push(ts);
        self.expecting_end = false;
    }
}

/// An outlier detection algorithm.
pub trait OutlierDetector<'a> {
    /// The preprocessed data used by the outlier detection algorithm.
    ///
    /// This type is used to store the preprocessed data that is
    /// calculated from the input data. The preprocessed data is
    /// then used by the `detect` method to determine whether each
    /// series is an outlier.
    ///
    /// An example of preprocessed data might be the transposed
    /// input data, where elements of the inner vectors correspond
    /// to the same timestamp in different series.
    type PreprocessedData;

    /// Preprocess the given slice of series.
    ///
    /// The input is a slice of aligned time series. Each series is
    /// a slice of `f64` which represents the values of the series
    /// over time. The length of the inner slice is the same for all series.
    ///
    /// The output is a preprocessed version of the input data. The exact
    /// format of the preprocessed data is up to the implementation.
    /// The preprocessed data will be passed to the `detect` method.
    ///
    /// This method is separate from `detect` to allow for more efficient
    /// recalculations of the preprocessed data when some input parameters
    /// change. For example, if the input data is the same but the sensitivity
    /// changes, the outlier detection calculation can be rerun without
    /// reprocessing the input data.
    fn preprocess(&self, y: &'a [&'a [f64]]) -> Self::PreprocessedData;

    /// Detect outliers in the given slice of series.
    ///
    /// The output is a vector of `Series` where each `Series` corresponds
    /// to the corresponding series in the input. The implementation will
    /// decide whether each series is an outlier, i.e. whether it behaves
    /// differently to the other input series.
    fn detect(&self, y: &'a Self::PreprocessedData) -> OutlierResult;
}

// fn transpose(data: &[&[f64]]) -> Vec<Vec<f64>> {
//     let mut transposed = vec![vec![]; data.len()];
//     for row in data {
//         transposed.reserve(data.len());
//         for (i, value) in row.iter().enumerate() {
//             transposed[i].push(*value);
//         }
//     }
//     transposed
// }

// #[cfg(test)]
// mod test {
//     use super::*;

//     struct DummyDetector;

//     impl OutlierDetector for DummyDetector {
//         fn detect(&self, y: &[InputSeries<'_>]) -> OutlierResult {
//             let serieses = y
//                 .iter()
//                 .map(|series| {
//                     let is_outlier = series.iter().any(|&x| x > 10.0);
//                     let scores = series.to_vec();
//                     Series::new(is_outlier, scores)
//                 })
//                 .collect();
//             let band = Band {
//                 min: vec![-1.0; y[0].len()],
//                 max: vec![1.0; y[0].len()],
//             };
//             OutlierResult::new(serieses, band)
//         }
//     }
// }
