/// A positive-only, 64 bit precision floating point number.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PositiveFloat(f64);

/// An invalid float was provided when trying to create a [`PositiveFloat`].
#[derive(Debug, thiserror::Error)]
#[error("negative float provided: {0}")]
pub struct NegativeFloatError(f64);

impl PositiveFloat {
    /// Attempt to create a new `PositiveFloat`.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided float is less than 0.0.
    pub fn try_new(f: f64) -> Result<Self, NegativeFloatError> {
        if f <= 0.0 {
            return Err(NegativeFloatError(f));
        }
        Ok(Self(f))
    }

    /// Create a new `PositiveFloat` with the value 1.0.
    pub const fn one() -> Self {
        Self(1.0)
    }
}

impl TryFrom<f64> for PositiveFloat {
    type Error = NegativeFloatError;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

impl std::ops::Deref for PositiveFloat {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<PositiveFloat> for f64 {
    fn from(value: PositiveFloat) -> Self {
        value.0
    }
}
