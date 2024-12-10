#![doc = include_str!("../README.md")]
mod data;
mod error;
mod features;
pub mod forecaster;
// Export the optimizer module so that users can implement their own
// optimizers.
pub mod optimizer;
mod positive_float;
mod prophet;
#[cfg(test)]
mod testdata;
mod util;

#[cfg(feature = "cmdstan")]
pub mod cmdstan;
#[cfg(feature = "wasmstan-min")]
pub mod wasmstan;

/// A timestamp represented as seconds since the epoch.
pub type TimestampSeconds = i64;

// Re-export everything at the root so that users don't have to
// navigate the module hierarchy.
pub use data::{PredictionData, TrainingData};
pub use error::Error;
pub use features::{FeatureMode, Holiday, HolidayOccurrence, Regressor, Seasonality, Standardize};
pub use optimizer::{Algorithm, Optimizer, TrendIndicator};
pub use positive_float::{PositiveFloat, TryFromFloatError};
pub use prophet::{
    options::{
        EstimationMode, GrowthType, IntervalWidth, OptProphetOptions, ProphetOptions, Scaling,
        SeasonalityOption,
    },
    predict::{FeaturePrediction, IncludeHistory, Predictions},
    Prophet,
};
