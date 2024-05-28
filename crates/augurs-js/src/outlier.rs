use std::collections::HashSet;

use augurs_outlier::OutlierDetector as _;
use js_sys::Float64Array;
use serde::{Deserialize, Serialize};
use tsify::Tsify;

use wasm_bindgen::prelude::*;

/// Options for the DBSCAN outlier detector.
#[derive(Debug, Default, Deserialize, Tsify)]
#[tsify(from_wasm_abi)]
pub struct DBSCANDetectorOptions {
    /// A scale-invariant sensitivity parameter.
    ///
    /// This must be in (0, 1) and will be used to estimate a sensible
    /// value of epsilon based on the data.
    pub sensitivity: f64,
}

#[derive(Debug)]
enum Detector {
    Dbscan(augurs_outlier::DBSCANDetector),
}

impl Detector {
    fn detect(&self, y: Float64Array, n_timestamps: usize) -> OutlierResult {
        match self {
            Self::Dbscan(detector) => {
                let vec = y.to_vec();
                let y: Vec<_> = vec.chunks(n_timestamps).map(Into::into).collect();
                let result = detector.detect(&y);
                OutlierResult {
                    outlying_series: result.outlying_series,
                    series_results: result.series_results.into_iter().map(Into::into).collect(),
                    cluster_band: result.cluster_band.into(),
                }
            }
        }
    }
}

/// A detector for detecting outlying time series in a group of series.
#[derive(Debug)]
#[wasm_bindgen]
pub struct OutlierDetector {
    // Hide the internal detector type from the public API.
    detector: Detector,
}

#[wasm_bindgen]
impl OutlierDetector {
    /// Create a new outlier detector using the DBSCAN algorithm.
    #[wasm_bindgen]
    pub fn dbscan(options: DBSCANDetectorOptions) -> Result<OutlierDetector, JsError> {
        Ok(Self {
            detector: Detector::Dbscan(augurs_outlier::DBSCANDetector::with_sensitivity(
                options.sensitivity,
            )?),
        })
    }

    /// Detect outlying time series in a group of series.
    #[wasm_bindgen]
    pub fn detect(&self, y: Float64Array, n_timestamps: usize) -> OutlierResult {
        self.detector.detect(y, n_timestamps)
    }
}

/// A band indicating the min and max value considered outlying
/// at each timestamp.
#[derive(Debug, Clone, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
struct ClusterBand {
    /// The minimum value considered outlying at each timestamp.
    min: Vec<f64>,
    /// The maximum value considered outlying at each timestamp.
    max: Vec<f64>,
}

impl From<augurs_outlier::Band> for ClusterBand {
    fn from(b: augurs_outlier::Band) -> Self {
        Self {
            min: b.min,
            max: b.max,
        }
    }
}

/// A potentially outlying series.
#[derive(Debug, Clone, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
struct OutlierSeries {
    /// Whether the series is an outlier for at least one of the samples.
    is_outlier: bool,
    /// The intervals of the series that are considered outliers.
    outlier_intervals: Vec<OutlierInterval>,
    /// The outlier scores of the series for each sample.
    scores: Vec<f64>,
}

impl From<augurs_outlier::Series> for OutlierSeries {
    fn from(s: augurs_outlier::Series) -> Self {
        Self {
            is_outlier: s.is_outlier,
            outlier_intervals: convert_intervals(s.outlier_intervals),
            scores: s.scores,
        }
    }
}

/// An interval for which a series is outlying.
#[derive(Debug, Clone, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
struct OutlierInterval {
    /// The start index of the interval.
    start: usize,
    /// The end index of the interval, if any.
    end: Option<usize>,
}

fn convert_intervals(intervals: augurs_outlier::OutlierIntervals) -> Vec<OutlierInterval> {
    let mut out = Vec::with_capacity(intervals.indices.len() / 2);
    if intervals.indices.is_empty() {
        return out;
    }
    for chunk in intervals.indices.chunks(2) {
        out.push(OutlierInterval {
            start: chunk[0],
            end: chunk.get(1).copied(),
        });
    }
    out
}

/// The result of applying an outlier detection algorithm to a group of time series.
#[derive(Debug, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
pub struct OutlierResult {
    /// The indexes of the series considered outliers.
    outlying_series: HashSet<usize>,
    /// The results of the detection for each series.
    series_results: Vec<OutlierSeries>,
    /// The band indicating the min and max value considered outlying
    /// at each timestamp.
    cluster_band: ClusterBand,
}
