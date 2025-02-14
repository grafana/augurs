//! JS bindings for outlier detection algorithms.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_core_js::VecVecF64;
use augurs_outlier::OutlierDetector as _;

// Enums representing outlier detectors and 'loaded' outlier detectors
// (i.e. detectors that have already preprocessed some data and are
// ready to detect).

#[derive(Debug)]
enum Detector {
    Dbscan(augurs_outlier::DbscanDetector),
    Mad(augurs_outlier::MADDetector),
}

impl Detector {
    /// Preprocess the data for the detector.
    ///
    /// This is provided as a separate method to allow for the
    /// preprocessed data to be cached in the future.
    fn preprocess(&self, series: &[Vec<f64>]) -> Result<LoadedDetector, JsError> {
        let series: Vec<_> = series.iter().map(|x| x.as_slice()).collect();
        match self {
            Self::Dbscan(detector) => {
                let data = augurs_outlier::DbscanDetector::preprocess(&series)?;
                Ok(LoadedDetector::Dbscan {
                    detector: detector.clone(),
                    data,
                })
            }
            Self::Mad(detector) => {
                let data = augurs_outlier::MADDetector::preprocess(&series)?;
                Ok(LoadedDetector::Mad {
                    detector: detector.clone(),
                    data,
                })
            }
        }
    }

    /// Preprocess and perform outlier detection on the data.
    fn detect(&self, series: &[Vec<f64>]) -> Result<OutlierOutput, JsError> {
        let series: Vec<_> = series.iter().map(|x| x.as_slice()).collect();
        match self {
            Self::Dbscan(detector) => {
                let data = augurs_outlier::DbscanDetector::preprocess(&series)?;
                Ok(detector.detect(&data)?.into())
            }
            Self::Mad(detector) => {
                let data = augurs_outlier::MADDetector::preprocess(&series)?;
                Ok(detector.detect(&data)?.into())
            }
        }
    }
}

#[derive(Debug)]
enum LoadedDetector {
    Dbscan {
        detector: augurs_outlier::DbscanDetector,
        data: <augurs_outlier::DbscanDetector as augurs_outlier::OutlierDetector>::PreprocessedData,
    },
    Mad {
        detector: augurs_outlier::MADDetector,
        data: <augurs_outlier::MADDetector as augurs_outlier::OutlierDetector>::PreprocessedData,
    },
}

impl LoadedDetector {
    fn detect(&self) -> Result<augurs_outlier::OutlierOutput, augurs_outlier::Error> {
        match self {
            Self::Dbscan { detector, data } => detector.detect(data),
            Self::Mad { detector, data } => detector.detect(data),
        }
    }
}

// The public API for the outlier detector, exposed via the Javascript bindings.

/// Options for the DBSCAN outlier detector.
#[derive(Debug, Default, Deserialize, Tsify)]
#[tsify(from_wasm_abi)]
pub struct OutlierDetectorOptions {
    /// A scale-invariant sensitivity parameter.
    ///
    /// This must be in (0, 1) and will be used to estimate a sensible
    /// value of epsilon based on the data.
    pub sensitivity: f64,
}

/// The type of outlier detector to use.
#[derive(Debug, Clone, Copy, Deserialize, Tsify)]
#[serde(rename_all = "lowercase")]
#[tsify(from_wasm_abi)]
pub enum OutlierDetectorType {
    /// A DBSCAN-based outlier detector.
    Dbscan,
    /// A MAD-based outlier detector.
    Mad,
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
    /// Create a new outlier detector.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[allow(non_snake_case)] detectorType: OutlierDetectorType,
        options: OutlierDetectorOptions,
    ) -> Result<OutlierDetector, JsError> {
        match detectorType {
            OutlierDetectorType::Dbscan => Self::dbscan(options),
            OutlierDetectorType::Mad => Self::mad(options),
        }
    }

    /// Create a new outlier detector using the DBSCAN algorithm.
    #[wasm_bindgen]
    pub fn dbscan(options: OutlierDetectorOptions) -> Result<OutlierDetector, JsError> {
        Ok(Self {
            detector: Detector::Dbscan(augurs_outlier::DbscanDetector::with_sensitivity(
                options.sensitivity,
            )?),
        })
    }

    /// Create a new outlier detector using the MAD algorithm.
    pub fn mad(options: OutlierDetectorOptions) -> Result<OutlierDetector, JsError> {
        Ok(Self {
            detector: Detector::Mad(augurs_outlier::MADDetector::with_sensitivity(
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
    pub fn detect(&self, y: VecVecF64) -> Result<OutlierOutput, JsError> {
        self.detector.detect(&y.convert()?)
    }

    /// Preprocess the data for the detector.
    ///
    /// The returned value is a 'loaded' outlier detector, which can be used
    /// to detect outliers without needing to preprocess the data again.
    ///
    /// This is useful if you plan to run the detector multiple times on the same data.
    #[wasm_bindgen]
    pub fn preprocess(&self, y: VecVecF64) -> Result<LoadedOutlierDetector, JsError> {
        Ok(LoadedOutlierDetector {
            detector: self.detector.preprocess(&y.convert()?)?,
        })
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
    /// Detect outliers in the given time series.
    #[wasm_bindgen]
    pub fn detect(&self) -> Result<OutlierOutput, JsError> {
        Ok(self.detector.detect()?.into())
    }

    /// Update the detector with new options.
    ///
    /// # Errors
    ///
    /// This method will return an error if the detector and options types
    /// are incompatible.
    #[wasm_bindgen(js_name = "updateDetector")]
    pub fn update_detector(&mut self, options: OutlierDetectorOptions) -> Result<(), JsError> {
        match &mut self.detector {
            LoadedDetector::Dbscan {
                ref mut detector, ..
            } => {
                // This isn't ideal because it doesn't maintain any other state of the detector,
                // but it's the best we can do without adding an `update` method to the `OutlierDetector`
                // trait, which would in turn require some sort of config associated type.
                let _ = std::mem::replace(
                    detector,
                    augurs_outlier::DbscanDetector::with_sensitivity(options.sensitivity)?,
                );
            }
            LoadedDetector::Mad {
                ref mut detector, ..
            } => {
                // This isn't ideal because it doesn't maintain any other state of the detector,
                // but it's the best we can do without adding an `update` method to the `OutlierDetector`
                // trait, which would in turn require some sort of config associated type.
                let _ = std::mem::replace(
                    detector,
                    augurs_outlier::MADDetector::with_sensitivity(options.sensitivity)?,
                );
            }
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
            outlier_intervals: s
                .outlier_intervals
                .intervals
                .into_iter()
                .map(Into::into)
                .collect(),
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

impl From<augurs_outlier::OutlierInterval> for OutlierInterval {
    fn from(i: augurs_outlier::OutlierInterval) -> Self {
        Self {
            start: i.start,
            end: i.end,
        }
    }
}

/// The result of applying an outlier detection algorithm to a group of time series.
#[derive(Debug, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
pub struct OutlierOutput {
    /// The indexes of the series considered outliers.
    outlying_series: BTreeSet<usize>,
    /// The results of the detection for each series.
    series_results: Vec<OutlierSeries>,
    /// The band indicating the min and max value considered outlying
    /// at each timestamp.
    ///
    /// This may be undefined if no cluster was found (for example if
    /// there were fewer than 3 series in the input data in the case of
    /// DBSCAN).
    cluster_band: Option<ClusterBand>,
}

impl From<augurs_outlier::OutlierOutput> for OutlierOutput {
    fn from(r: augurs_outlier::OutlierOutput) -> Self {
        Self {
            outlying_series: r.outlying_series,
            series_results: r.series_results.into_iter().map(Into::into).collect(),
            cluster_band: r.cluster_band.map(Into::into),
        }
    }
}
