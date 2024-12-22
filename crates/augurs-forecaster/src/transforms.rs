/*!
Data transformations.
*/

// Note: implementations of the various transforms are in the
// various submodules of this module (e.g. `power` and `scale`).

mod error;
mod exp;
pub mod interpolate;
mod power;
mod scale;

use std::fmt;

use augurs_core::Forecast;

pub use error::Error;
pub use exp::{Log, Logit};
pub use interpolate::{InterpolateExt, LinearInterpolator};
pub use power::{BoxCox, YeoJohnson};
pub use scale::{MinMaxScaler, StandardScaleParams, StandardScaler};

/// A transformation pipeline.
///
/// A `Pipeline` is a collection of heterogeneous [`Transformer`] instances
/// that can be applied to a time series. Calling [`Pipeline::fit`] or [`Pipeline::fit_transform`]
/// will fit each transformation to the output of the previous one in turn
/// starting by passing the input to the first transformation. The
/// [`Pipeline::inverse_transform`] can then be used to back-transform data
/// to the original scale.
#[derive(Debug, Default)]
pub struct Pipeline {
    transformers: Vec<Box<dyn Transformer>>,
    is_fitted: bool,
}

impl Pipeline {
    /// Create a new `Pipeline` with the given transformers.
    pub fn new(transformers: Vec<Box<dyn Transformer>>) -> Self {
        Self {
            transformers,
            is_fitted: false,
        }
    }

    // Helper function for actually doing the fit then transform steps.
    fn fit_transform_inner(&mut self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.transformers.iter_mut() {
            t.fit_transform(input)?;
        }
        self.is_fitted = true;
        Ok(())
    }

    /// Apply the inverse transformations to the given forecast.
    ///
    /// # Errors
    ///
    /// This function will return an error if the pipeline has not been fitted.
    pub(crate) fn inverse_transform_forecast(&self, forecast: &mut Forecast) -> Result<(), Error> {
        for t in self.transformers.iter().rev() {
            t.inverse_transform(&mut forecast.point)?;
            if let Some(intervals) = forecast.intervals.as_mut() {
                t.inverse_transform(&mut intervals.lower)?;
                t.inverse_transform(&mut intervals.upper)?;
            }
        }
        Ok(())
    }
}

impl Transformer for Pipeline {
    /// Fit the transformations to the given time series.
    ///
    /// Prefer `fit_transform` if possible, as it avoids copying the input.
    fn fit(&mut self, input: &[f64]) -> Result<(), Error> {
        // Copy the input to avoid mutating the original.
        // We need to do this so we can call `fit_transform` on each
        // transformation in the pipeline without mutating the input.
        // This is required because each transformation needs to be
        // fit after previous transformations have been applied.
        let mut input = input.to_vec();
        // Reuse `fit_transform_inner`, and just discard the result.
        self.fit_transform_inner(&mut input)?;
        Ok(())
    }

    /// Fit and transform the given time series.
    ///
    /// This is equivalent to calling `fit` and then `transform` on the pipeline,
    /// but is more efficient because it avoids copying the input.
    fn fit_transform(&mut self, input: &mut [f64]) -> Result<(), Error> {
        self.fit_transform_inner(input)?;
        Ok(())
    }

    /// Apply the fitted transformations to the given time series.
    ///
    /// # Errors
    ///
    /// This function will return an error if the pipeline has not been fitted.
    fn transform(&self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.transformers.iter() {
            t.transform(input)?;
        }
        Ok(())
    }

    /// Apply the inverse transformations to the given time series.
    ///
    /// # Errors
    ///
    /// This function will return an error if the pipeline has not been fitted.
    fn inverse_transform(&self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.transformers.iter().rev() {
            t.inverse_transform(input)?;
        }
        Ok(())
    }
}

/// A transformation that can be applied to a time series.
pub trait Transformer: fmt::Debug + Sync + Send {
    /// Fit the transformation to the given time series.
    ///
    /// For example, for a min-max scaler, this would find
    /// the min and max of the provided data and store it on the
    /// scaler ready for use in transforming and back-transforming.
    fn fit(&mut self, data: &[f64]) -> Result<(), Error>;

    /// Apply the transformation to the given time series.
    ///
    /// # Errors
    ///
    /// This function should return an error if the transform has not been fitted,
    /// and may return other errors specific to the implementation.
    fn transform(&self, data: &mut [f64]) -> Result<(), Error>;

    /// Apply the inverse transformation to the given time series.
    ///
    /// # Errors
    ///
    /// This function should return an error if the transform has not been fitted,
    /// and may return other errors specific to the implementation.
    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error>;

    /// Fit the transformation to the given time series and then apply it.
    ///
    /// The default implementation just calls [`Self::fit`] then [`Self::transform`]
    /// but it can be overridden to be more efficient if desired.
    fn fit_transform(&mut self, data: &mut [f64]) -> Result<(), Error> {
        self.fit(data)?;
        self.transform(data)?;
        Ok(())
    }

    /// Create a boxed version of the transformation.
    ///
    /// This is useful for creating a `Transform` instance that can be used as
    /// part of a [`Pipeline`].
    fn boxed(self) -> Box<dyn Transformer>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}
