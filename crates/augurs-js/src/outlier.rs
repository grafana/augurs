use std::collections::HashSet;

use augurs_outlier::OutlierDetector as _;
use js_sys::Float64Array;
use serde::{Deserialize, Serialize};
use tsify::Tsify;

use wasm_bindgen::prelude::*;

// Enums representing outlier detectors and 'loaded' outlier detectors
// (i.e. detectors that have already preprocessed some data and are
// ready to detect).

#[derive(Debug)]
enum Detector {
    Dbscan(augurs_outlier::DBSCANDetector),
}

impl Detector {
    /// Preprocess the data for the detector.
    ///
    /// This is provided as a separate method to allow for the
    /// preprocessed data to be cached in the future.
    fn preprocess(&self, y: Float64Array, nTimestamps: usize) -> LoadedDetector {
        match self {
            Self::Dbscan(detector) => {
                let vec = y.to_vec();
                let y: Vec<_> = vec.chunks(nTimestamps).map(Into::into).collect();
                let data = detector.preprocess(&y);
                LoadedDetector::Dbscan {
                    detector: detector.clone(),
                    data,
                }
            }
        }
    }

    /// Preprocess and perform outlier detection on the data.
    fn detect(&self, y: Float64Array, nTimestamps: usize) -> OutlierResult {
        match self {
            Self::Dbscan(detector) => {
                let vec = y.to_vec();
                let y: Vec<_> = vec.chunks(nTimestamps).map(Into::into).collect();
                let data = detector.preprocess(&y);
                detector.detect(&data).into()
            }
        }
    }
}

#[derive(Debug)]
enum LoadedDetector {
    Dbscan {
        detector: augurs_outlier::DBSCANDetector,
        data: <augurs_outlier::DBSCANDetector as augurs_outlier::OutlierDetector>::PreprocessedData,
    },
}

impl LoadedDetector {
    fn detect(&self) -> augurs_outlier::OutlierResult {
        match self {
            Self::Dbscan { detector, data } => detector.detect(data),
        }
    }
}

// The public API for the outlier detector, exposed via the Javascript bindings.

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

#[derive(Debug, Default, Deserialize, Tsify)]
#[tsify(from_wasm_abi)]
pub struct MADDetectorOptions {
    /// A scale-invariant sensitivity parameter.
    ///
    /// This must be in (0, 1) and will be used to estimate a sensible
    /// value of epsilon based on the data.
    pub sensitivity: f64,
}

#[derive(Debug, Deserialize, Tsify)]
#[tsify(from_wasm_abi)]
/// Options for outlier detectors.
pub enum OutlierDetectorOptions {
    #[serde(rename = "dbscan")]
    Dbscan(DBSCANDetectorOptions),
    #[serde(rename = "mad")]
    Mad(MADDetectorOptions),
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
    ///
    /// Note: if you plan to run the detector multiple times on the same data,
    /// you should use the `preprocess` method to cache the preprocessed data,
    /// then call `detect` on the `LoadedOutlierDetector` returned by `preprocess`.
    #[wasm_bindgen]
    pub fn detect(&self, y: Float64Array, n_timestamps: usize) -> OutlierResult {
        self.detector.detect(y, n_timestamps)
    }

    /// Preprocess the data for the detector.
    ///
    /// The returned value is a 'loaded' outlier detector, which can be used
    /// to detect outliers without needing to preprocess the data again.
    ///
    /// This is useful if you plan to run the detector multiple times on the same data.
    #[wasm_bindgen]
    pub fn preprocess(&self, y: Float64Array, n_timestamps: usize) -> LoadedOutlierDetector {
        LoadedOutlierDetector {
            detector: self.detector.preprocess(y, n_timestamps),
        }
    }
}

/// A 'loaded' outlier detector, ready to detect outliers.
///
/// This is returned by the `preprocess` method of `OutlierDetector`,
/// and holds the preprocessed data for the detector.
#[derive(Debug)]
#[wasm_bindgen]
pub struct LoadedOutlierDetector {
    detector: LoadedDetector,
}

#[wasm_bindgen]
impl LoadedOutlierDetector {
    #[wasm_bindgen]
    pub fn detect(&self) -> OutlierResult {
        self.detector.detect().into()
    }

    /// Update the detector with new options.
    ///
    /// # Errors
    ///
    /// This method will return an error if the detector and options types
    /// are incompatible.
    #[wasm_bindgen(js_name = "updateDetector")]
    pub fn update_detector(&mut self, options: OutlierDetectorOptions) -> Result<(), JsError> {
        match (&mut self.detector, options) {
            (
                LoadedDetector::Dbscan {
                    ref mut detector, ..
                },
                OutlierDetectorOptions::Dbscan(options),
            ) => {
                // This isn't ideal because it doesn't maintain any other state of the detector,
                // but it's the best we can do without adding an `update` method to the `OutlierDetector`
                // trait, which would in turn require some sort of config associated type.
                let _ = std::mem::replace(
                    detector,
                    augurs_outlier::DBSCANDetector::with_sensitivity(options.sensitivity)?,
                );
            }
            _ => return Err(JsError::new("Mismatch between detector and options")),
        }
        Ok(())
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

impl From<augurs_outlier::OutlierResult> for OutlierResult {
    fn from(r: augurs_outlier::OutlierResult) -> Self {
        Self {
            outlying_series: r.outlying_series,
            series_results: r.series_results.into_iter().map(Into::into).collect(),
            cluster_band: r.cluster_band.into(),
        }
    }
}
