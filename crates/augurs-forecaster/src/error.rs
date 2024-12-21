use augurs_core::ModelError;

use crate::transforms;

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The model has not yet been fit.
    #[error("Model not yet fit")]
    ModelNotYetFit,
    /// An error occurred while fitting a model.
    #[error("Fit error: {source}")]
    Fit {
        /// The original error.
        source: Box<dyn ModelError>,
    },
    /// An error occurred while making predictions for a model.
    #[error("Predict error: {source}")]
    Predict {
        /// The original error.
        source: Box<dyn ModelError>,
    },

    /// An error occurred while running a transformation.
    #[error("Transform error: {source}")]
    Transform {
        /// The original error.
        #[from]
        source: transforms::Error,
    },
}
