//! Prophet is a re-implementation of the Prophet forecasting algorithm.
mod data;
mod error;
mod features;
// Export the optimizer module so that users can implement their own
// optimizers.
pub mod optimizer;
mod positive_float;
mod prophet;

pub type TimestampSeconds = u64;

// Re-export everything at the root so that users don't have to
// navigate the module hierarchy.
pub use data::TrainingData;
pub use error::Error;
pub use features::{FeatureMode, Holiday, Regressor, Seasonality, Standardize};
pub use optimizer::{Algorithm, Optimizer, TrendIndicator};
pub use positive_float::PositiveFloat;
pub use prophet::{
    options::{EstimationMode, GrowthType, ProphetOptions, Scaling, SeasonalityOption},
    Prophet,
};
