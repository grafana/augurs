#![doc = include_str!("../README.md")]

use std::collections::BTreeSet;

mod dbscan;
mod error;
mod mad;
mod sensitivity;
#[cfg(test)]
mod testing;

pub use dbscan::{Data as DbscanData, DbscanDetector};
pub use error::Error;
pub use mad::{MADDetector, PreprocessedData as MADData};
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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct OutlierOutput {
    /// The indexes of the series considered outliers.
    ///
    /// This is a `BTreeSet` to ensure that the order of the series is preserved.
    pub outlying_series: BTreeSet<usize>,

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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
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
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", transparent))]
pub struct OutlierIntervals {
    /// The list of outlier intervals.
    pub intervals: Vec<OutlierInterval>,

    /// Are we expecting a start or end timestamp to be pushed next?
    #[cfg_attr(feature = "serde", serde(skip))]
    expecting_end: bool,
}

impl OutlierIntervals {
    fn empty() -> Self {
        Self {
            intervals: Vec::new(),
            expecting_end: false,
        }
    }

    fn add_start(&mut self, ts: usize) {
        debug_assert!(
            !self.expecting_end,
            "Expected end of outlier interval, got start"
        );

        self.intervals.push(OutlierInterval {
            start: ts,
            end: None,
        });
        self.expecting_end = true;
    }

    fn add_end(&mut self, ts: usize) {
        debug_assert!(
            self.expecting_end,
            "Expected start of outlier interval, got end"
        );

        match self.intervals.last_mut() {
            Some(x @ OutlierInterval { end: None, .. }) => {
                x.end = Some(ts);
            }
            _ => unreachable!("tried to add end to an open-ended interval"),
        };
        self.expecting_end = false;
    }
}

/// A single outlier interval.
///
/// An outlier interval is a contiguous range of indices in a time series
/// where an outlier is detected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct OutlierInterval {
    /// The start index of the interval.
    pub start: usize,
    /// The end index of the interval, if it exists.
    ///
    /// If the interval is open-ended, this will be `None`.
    pub end: Option<usize>,
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

#[cfg(test)]
mod test {
    use super::*;

    struct DummyDetector;

    impl OutlierDetector for DummyDetector {
        type PreprocessedData = Vec<Vec<f64>>;

        fn preprocess(&self, y: &[&[f64]]) -> Result<Self::PreprocessedData, Error> {
            Ok(y.iter().map(|x| x.to_vec()).collect())
        }

        fn detect(&self, y: &Self::PreprocessedData) -> Result<OutlierOutput, Error> {
            let serieses = y
                .iter()
                .map(|series| {
                    let mut intervals = OutlierIntervals::empty();
                    intervals.add_start(1);
                    Series {
                        is_outlier: series.iter().any(|&x| x > 10.0),
                        scores: series.to_vec(),
                        outlier_intervals: intervals,
                    }
                })
                .collect();
            let band = Band {
                min: vec![-1.0; y[0].len()],
                max: vec![1.0; y[0].len()],
            };
            Ok(OutlierOutput::new(serieses, Some(band)))
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialize() {
        let mut outlier_intervals = OutlierIntervals::empty();
        outlier_intervals.add_start(1);
        let series = Series {
            is_outlier: true,
            scores: vec![1.0, 2.0, 3.0],
            outlier_intervals,
        };
        let output = OutlierOutput {
            outlying_series: BTreeSet::from([0, 1]),
            series_results: vec![series],
            cluster_band: None,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        assert_eq!(
            serialized,
            r#"{"outlyingSeries":[0,1],"seriesResults":[{"isOutlier":true,"outlierIntervals":[{"start":1,"end":null}],"scores":[1.0,2.0,3.0]}],"clusterBand":null}"#
        );
    }
}
