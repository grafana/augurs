use crate::sensitivity::SensitivityError;

/// Errors that can occur when working with outlier detection algorithms.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error occurred during sensitivity calculation.
    ///
    /// This probably means the provided sensitivity value was out of bounds.
    #[error("sensitivity: {0}")]
    Sensitivity(#[from] SensitivityError),

    /// An error occurred during preprocessing.
    #[error("preprocessing: {0}")]
    Preprocessing(#[from] PreprocessingError),

    /// An error occurred during detection.
    #[error("detecting: {0}")]
    Detection(#[from] DetectionError),
}

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct PreprocessingError {
    #[from]
    source: Box<dyn std::error::Error>,
}

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct DetectionError {
    #[from]
    source: Box<dyn std::error::Error>,
}
