#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

use std::collections::HashSet;

mod dbscan;
mod error;
mod mad;
mod sensitivity;
#[cfg(test)]
mod testing;

pub use dbscan::DBSCANDetector;
pub use error::Error;
pub use mad::MADDetector;
use sensitivity::Sensitivity;

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
pub struct OutlierOutput {
    /// The indexes of the series considered outliers.
    pub outlying_series: HashSet<usize>,

    /// The results of the detection for each series.
    pub series_results: Vec<Series>,

    /// The band indicating the min and max value considered outlying
    /// at each timestamp.
    ///
    /// This may be `None` if no cluster was found (for example if
    /// there were fewer than 3 series in the input data in the case of
    /// DBSCAN).
    pub cluster_band: Option<Band>,
}

impl OutlierOutput {
    /// Create a new `OutlierResult` from the given series results.
    pub fn new(series_results: Vec<Series>, cluster_band: Option<Band>) -> Self {
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

    /// Create a `Vec<Series>` with length `n_series` where each inner `Series`
    /// has its scores preallocated to length `n_timestamps`, all initialized to 0.0.
    pub fn preallocated(n_series: usize, n_timestamps: usize) -> Vec<Self> {
        std::iter::repeat_with(|| {
            let mut s = Series::with_capacity(n_timestamps);
            s.scores.resize(n_timestamps, 0.0);
            s
        })
        .take(n_series)
        .collect()
    }
}

/// A list of outlier intervals for a single series.
// We manually implement [`Serialize`] for this struct, serializing
// just the `timestamps` field as an array.
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
pub trait OutlierDetector {
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
    fn preprocess(&self, y: &[&[f64]]) -> Result<Self::PreprocessedData, Error>;

    /// Detect outliers in the given slice of series.
    ///
    /// The output is a vector of `Series` where each `Series` corresponds
    /// to the corresponding series in the input. The implementation will
    /// decide whether each series is an outlier, i.e. whether it behaves
    /// differently to the other input series.
    fn detect(&self, y: &Self::PreprocessedData) -> Result<OutlierOutput, Error>;
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
