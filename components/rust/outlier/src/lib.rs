#![doc = include_str!("../README.md")]

use std::fmt;

use augurs_outlier::{DbscanDetector, MADDetector, OutlierDetector};
use getrandom::Error;

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
use bindings::{
    augurs::outlier::types::{
        Algorithm, Band, EpsilonOrSensitivity, Input, OutlierInterval, Output, Series,
        ThresholdOrSensitivity,
    },
    export, Guest,
};

struct OutlierWorld;
export!(OutlierWorld);

impl Guest for OutlierWorld {
    fn detect(input: Input) -> Result<Output, String> {
        detect(input).map_err(|e| e.to_string())
    }
}

#[derive(Debug)]
enum TypedError {
    InvalidSensitivity(augurs_outlier::Error),
    TryFromIntError(std::num::TryFromIntError),
}

impl From<augurs_outlier::Error> for TypedError {
    fn from(value: augurs_outlier::Error) -> Self {
        Self::InvalidSensitivity(value)
    }
}

impl From<std::num::TryFromIntError> for TypedError {
    fn from(value: std::num::TryFromIntError) -> Self {
        Self::TryFromIntError(value)
    }
}

impl fmt::Display for TypedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSensitivity(e) => write!(f, "invalid sensitivity: {}", e),
            Self::TryFromIntError(e) => write!(f, "overflow converting to u32: {}", e),
        }
    }
}

impl std::error::Error for TypedError {}

fn detect(input: Input) -> Result<Output, TypedError> {
    let data_ref: Vec<_> = input.data.iter().map(|v| v.as_slice()).collect();
    let output = match input.algorithm {
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
    Ok(Output {
        outlying_series: output
            .outlying_series
            .into_iter()
            .map(TryInto::<u32>::try_into)
            .collect::<Result<_, _>>()?,
        series_results: output
            .series_results
            .into_iter()
            .map(|series| {
                Ok(Series {
                    is_outlier: series.is_outlier,
                    outlier_intervals: series
                        .outlier_intervals
                        .intervals
                        .into_iter()
                        .map(|interval| {
                            Ok(OutlierInterval {
                                start: interval.start.try_into()?,
                                end: interval.end.map(TryInto::try_into).transpose()?,
                            })
                        })
                        .collect::<Result<_, TypedError>>()?,
                    scores: series.scores,
                })
            })
            .collect::<Result<_, TypedError>>()?,
        cluster_band: output.cluster_band.map(|band| Band {
            min: band.min,
            max: band.max,
        }),
    })
}
