/// An error that can occur during the transformation process.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error occurred during optimization.
    #[error("error in optimization: {0}")]
    Optimize(#[from] argmin::core::Error),
    /// No best parameter was found during optimization.
    #[error("no best parameter found")]
    NoBestParameter,
    /// The input data did not have a distinct minimum and maximum value.
    #[error("no min-max found: {0:?}")]
    MinMaxNotFound(itertools::MinMaxResult<f64>),
    /// The transform has not been fitted yet.
    #[error("transform has not been fitted yet")]
    NotFitted,
    /// The input data is empty.
    #[error("data must not be empty")]
    EmptyData,
    /// The input data contains non-positive values.
    #[error("data contains non-positive values")]
    NonPositiveData,
    /// The input values contain NaN.
    #[error("input values must not be NaN")]
    NaNValue,
    /// The input lambda must be finite.
    #[error("input lambda must be finite")]
    InvalidLambda,
    /// The variance must be positive.
    #[error("variance must be positive")]
    VarianceNotPositive,
    /// All data must be greater than 0.
    #[error("all data must be greater than 0")]
    AllDataNotPositive,
    /// The input data is not in the valid domain.
    #[error("invalid domain")]
    InvalidDomain,
}

impl From<itertools::MinMaxResult<f64>> for Error {
    fn from(e: itertools::MinMaxResult<f64>) -> Self {
        Self::MinMaxNotFound(e)
    }
}
