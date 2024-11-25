use crate::TimestampSeconds;

/// Errors that can occur when using the Prophet forecasting algorithm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The data provided had mismatched column lengths.
    #[error("Mismatched column lengths: {a_name} has length {a} but {b_name} has length {b}")]
    MismatchedLengths {
        /// The length of the first column.
        a: usize,
        /// The length of the second column.
        b: usize,
        /// The name of the first column.
        a_name: String,
        /// The name of the second column.
        b_name: String,
    },
    /// Not enough data was provided to fit the model.
    #[error("Not enough data")]
    NotEnoughData,
    /// Optimization failed for some reason.
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    /// An invalid interval width was passed.
    ///
    /// The interval width must be between 0.0 and 1.0.
    #[error("Invalid interval width: {0}; must be between 0.0 and 1.0")]
    InvalidIntervalWidth(f64),
    /// Too many data points were provided, overflowing an `i32`.
    #[error("Too many data points: {got}, max is {max}", got = .0, max = i32::MAX)]
    TooManyDataPoints(usize),
    /// The timestamps provided are constant.
    #[error("Timestamps are constant: {0}")]
    TimestampsAreConstant(TimestampSeconds),
    /// The same seasonality was added to the model twice.
    #[error("Attempted to add a seasonality with the same name: {0}")]
    DuplicateSeasonality(String),
    /// A column contained an infinite value.
    #[error("Found infinite value in column {column}")]
    InfiniteValue {
        /// The name of the column that contained an infinite value.
        column: String,
    },
    /// A column contained a NaN value.
    #[error("Found NaN value in column {column}")]
    NaNValue {
        /// The name of the column that contained a NaN value.
        column: String,
    },
    /// A regressor was added to the model but missing from the data.
    #[error("Missing regressor: {0}")]
    MissingRegressor(String),
    /// A seasonality in the model was marked as conditional but
    /// the condition column was missing from the data.
    #[error("Missing condition for seasonality: {0}")]
    MissingSeasonalityCondition(String),
    /// AbsMax scaling failed because the min and max were the same.
    #[error("AbsMax scaling failed because the min and max were the same")]
    AbsMaxScalingFailed,
    /// The `cap` column for logistic growth was missing.
    #[error("Missing cap for logistic growth")]
    MissingCap,
    /// The `floor` column for logistic growth was missing.
    #[error("Missing floor for logistic growth")]
    MissingFloor,
    /// The cap was not always greater than the floor.
    #[error("Cap is not greater than floor (which defaults to 0)")]
    CapNotGreaterThanFloor,
    /// One or more of the provided changepoints were out
    /// of the range of the training data.
    #[error("Changepoints must fall within training data")]
    ChangepointsOutOfRange,
    /// The Prophet model has not yet been fit.
    ///
    /// Fit the model first using [`Prophet::fit`](crate::Prophet::fit).
    #[error("Model has not been fit")]
    ModelNotFit,
    /// It was not possible to infer the frequency of the dates.
    ///
    /// This can happen if the dates are not evenly spaced, and
    /// there is no frequency that appears more often than others.
    #[error("Unable to infer frequency from dates: {0:?}")]
    UnableToInferFrequency(Vec<TimestampSeconds>),
}
