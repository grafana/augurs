/// A positive-only, 64 bit precision floating point number.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PositiveFloat(f64);

/// An invalid float was provided when trying to create a [`PositiveFloat`].
#[derive(Debug, thiserror::Error)]
#[error("invalid float provided: {0}")]
pub struct TryFromFloatError(f64);

impl PositiveFloat {
    /// Attempt to create a new `PositiveFloat`.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided float is not finite or less than or equal to 0.0.
    pub fn try_new(f: f64) -> Result<Self, TryFromFloatError> {
        if !f.is_finite() || f <= 0.0 {
            return Err(TryFromFloatError(f));
        }
        Ok(Self(f))
    }

    /// Create a new `PositiveFloat` with the value 1.0.
    pub const fn one() -> Self {
        Self(1.0)
    }
}

impl TryFrom<f64> for PositiveFloat {
    type Error = TryFromFloatError;
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
