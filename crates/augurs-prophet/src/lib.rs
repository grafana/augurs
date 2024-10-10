#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]
mod data;
mod error;
mod features;
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

/// A timestamp represented as seconds since the epoch.
pub type TimestampSeconds = i64;

// Re-export everything at the root so that users don't have to
// navigate the module hierarchy.
pub use data::{PredictionData, TrainingData};
pub use error::Error;
pub use features::{FeatureMode, Holiday, Regressor, Seasonality, Standardize};
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
