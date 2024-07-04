/// The sensitivity of an outlier detection algorithm.
///
/// Sensitivity values are between 0.0 and 1.0, where 0.0 means
/// the algorithm is not sensitive to outliers and 1.0 means the
/// algorithm is very sensitive to outliers.
///
/// The exact meaning of the sensitivity value depends on the
/// implementation of the outlier detection algorithm.
/// For example, a DBSCAN based algorithm might use the sensitivity
/// to determine the maximum distance between points in the same
/// cluster (i.e. `epsilon`).
///
/// Crucially, though, sensitivity will always be a value between 0.0
/// and 1.0 to make it easier to reason about for users.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Sensitivity(pub f64);

impl TryFrom<f64> for Sensitivity {
    type Error = SensitivityError;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value <= 0.0 || value >= 1.0 {
            Err(SensitivityError(value))
        } else {
            Ok(Self(value))
        }
    }
}

/// An error indicating that the sensitivity value is out of bounds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SensitivityError(f64);

impl std::fmt::Display for SensitivityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sensitivity must be between 0.0 and 1.0, got {}", self.0)
    }
}

impl std::error::Error for SensitivityError {}
