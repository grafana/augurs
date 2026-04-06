// TODO: promote this to augurs_core?

//! Custom error type for Augurs.
use thiserror::Error;

/// Error type for Augurs.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    /// The time series is too short for the requested ARIMA order.
    #[error("time series too short for ARIMA order: need {need}, got {got}")]
    SeriesTooShort {
        /// Number of data points needed.
        need: usize,
        /// Number of data points provided.
        got: usize,
    },

    /// An error occurred during model fitting.
    #[error("fitting error: {0}")]
    FitError(String),

    /// An error occurred during prediction.
    #[error("prediction error: {0}")]
    PredictionError(String),

    /// Invalid model parameters.
    #[error("invalid model parameters: {0}")]
    InvalidParameters(String),

    /// Mathematical error during computation.
    #[error("mathematical error: {0}")]
    MathError(String),
}

impl augurs_core::ModelError for Error {}

/// Result type
pub type Result<T> = std::result::Result<T, Error>;
