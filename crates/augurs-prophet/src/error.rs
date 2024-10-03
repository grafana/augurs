use crate::TimestampSeconds;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Mismatched column lengths: {a_name} has length {a} but {b_name} has length {b}")]
    MismatchedLengths {
        a: usize,
        b: usize,
        a_name: String,
        b_name: String,
    },
    #[error("Not enough data")]
    NotEnoughData,
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("Too many data points: {0}, max is {}", i32::MAX)]
    TooManyDataPoints(usize),
    #[error("Timestamps are constant: {0}")]
    TimestampsAreConstant(TimestampSeconds),
    #[error("Found infinite value in column {column}")]
    InfiniteValue { column: String },
    #[error("Missing regressor: {0}")]
    MissingRegressor(String),
    #[error("Missing condition for seasonality: {0}")]
    MissingSeasonalityCondition(String),
    #[error("Scaling failed")]
    Scaling,
    #[error("Missing cap for logistic growth")]
    MissingCap,
    #[error("Missing floor for logistic growth")]
    MissingFloor,
    #[error("Cap is not greater than floor (which defaults to 0)")]
    CapNotGreaterThanFloor,
    #[error("Prior scale must be greater than 0, got: {0}")]
    InvalidSeasonalityPriorScale(f64),
    #[error("Fourier order must be greater than 0, got: {0}")]
    InvalidFourierOrder(u32),
    #[error("Changepoints must fall within training data.")]
    ChangepointsOutOfRange,
}
