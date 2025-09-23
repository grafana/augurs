#![doc = include_str!("../README.md")]

use std::fmt;

use augurs_outlier::{DbscanDetector, MADDetector, OutlierDetector};
use getrandom::Error;
use serde::Deserialize;

// It looks like some dependency or other is importing getrandom despite not actually
// using it, so we can just provide a dummy implementation here.
// If we see errors later we could try to import a RNG source via the Component Model.
// See https://docs.rs/getrandom/latest/getrandom/#custom-backend for info on this.
#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(_dest: *mut u8, _len: usize) -> Result<(), Error> {
    Err(Error::UNSUPPORTED)
}

// Wrap the wit-bindgen macro in a module so we don't get warned about missing docs in the generated trait.
mod bindings {
    wit_bindgen::generate!({
        world: "outlier",
        default_bindings_module: "bindings",
    });
}
use bindings::{export, Guest};

struct OutlierWorld;
export!(OutlierWorld);

impl Guest for OutlierWorld {
    fn detect(input: String) -> Result<String, String> {
        detect(input).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged, rename_all = "camelCase")]
enum Algorithm {
    Dbscan(DbscanParams),
    Mad(MadParams),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DbscanParams {
    epsilon_or_sensitivity: EpsilonOrSensitivity,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
enum EpsilonOrSensitivity {
    Sensitivity(f64),
    Epsilon(f64),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MadParams {
    threshold_or_sensitivity: ThresholdOrSensitivity,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ThresholdOrSensitivity {
    Sensitivity(f64),
    Threshold(f64),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
struct Input {
    algorithm: Algorithm,
    data: Vec<Vec<f64>>,
}

#[derive(Debug)]
enum TypedError {
    InvalidInputJSON(serde_json::Error),
    InvalidSensitivity(augurs_outlier::Error),
    InvalidOutputJSON(serde_json::Error),
}

impl From<augurs_outlier::Error> for TypedError {
    fn from(value: augurs_outlier::Error) -> Self {
        Self::InvalidSensitivity(value)
    }
}

impl fmt::Display for TypedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInputJSON(e) => write!(f, "invalid input JSON: {}", e),
            Self::InvalidSensitivity(e) => write!(f, "invalid sensitivity: {}", e),
            Self::InvalidOutputJSON(e) => write!(f, "invalid output JSON: {}", e),
        }
    }
}

impl std::error::Error for TypedError {}

fn detect(input: String) -> Result<String, TypedError> {
    let Input { algorithm, data } =
        serde_json::from_str(&input).map_err(TypedError::InvalidInputJSON)?;
    let data_ref: Vec<_> = data.iter().map(|v| v.as_slice()).collect();
    let output = match algorithm {
        Algorithm::Dbscan(params) => {
            let detector = match params.epsilon_or_sensitivity {
                EpsilonOrSensitivity::Sensitivity(s) => DbscanDetector::with_sensitivity(s)?,
                EpsilonOrSensitivity::Epsilon(e) => DbscanDetector::with_epsilon(e),
            };
            let preprocessed = detector.preprocess(&data_ref)?;
            detector.detect(&preprocessed)?
        }
        Algorithm::Mad(params) => {
            let detector = match params.threshold_or_sensitivity {
                ThresholdOrSensitivity::Sensitivity(s) => MADDetector::with_sensitivity(s)?,
                ThresholdOrSensitivity::Threshold(t) => MADDetector::with_threshold(t),
            };
            let preprocessed = detector.preprocess(&data_ref)?;
            detector.detect(&preprocessed)?
        }
    };
    serde_json::to_string(&output).map_err(TypedError::InvalidOutputJSON)
}
