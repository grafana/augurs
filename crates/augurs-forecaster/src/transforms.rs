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
/// The `Pipeline` struct is a collection of heterogeneous `Transform` instances
/// that can be applied to a time series. Calling the `fit` or `fit_transform`
/// methods will fit each transformation to the output of the previous one in turn
/// starting by passing the input to the first transformation.
#[derive(Debug, Default)]
pub struct Pipeline {
    transforms: Vec<Box<dyn Transform>>,
    is_fitted: bool,
}

impl Pipeline {
    /// Create a new `Pipeline` with the given transforms.
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self {
            transforms,
            is_fitted: false,
        }
    }

    /// Fit the transformations to the given time series.
    ///
    /// Prefer `fit_transform` if possible, as it avoids copying the input.
    pub fn fit(&mut self, input: &[f64]) -> Result<(), Error> {
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

    fn fit_transform_inner(&mut self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.transforms.iter_mut() {
            t.fit_transform(input)?;
        }
        self.is_fitted = true;
        Ok(())
    }

    /// Fit and transform the given time series.
    ///
    /// This is equivalent to calling `fit` and then `transform` on the pipeline,
    /// but is more efficient because it avoids copying the input.
    pub fn fit_transform(&mut self, input: &mut [f64]) -> Result<(), Error> {
        self.fit_transform_inner(input)?;
        Ok(())
    }

    /// Apply the fitted transformations to the given time series.
    ///
    /// # Errors
    ///
    /// This function will return an error if the pipeline has not been fitted.
    pub fn transform(&self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.transforms.iter() {
            t.transform(input)?;
        }
        Ok(())
    }

    /// Apply the inverse transformations to the given time series.
    ///
    /// # Errors
    ///
    /// This function will return an error if the pipeline has not been fitted.
    pub fn inverse_transform(&self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.transforms.iter().rev() {
            t.inverse_transform(input)?;
        }
        Ok(())
    }

    /// Apply the inverse transformations to the given forecast.
    ///
    /// # Errors
    ///
    /// This function will return an error if the pipeline has not been fitted.
    pub(crate) fn inverse_transform_forecast(&self, forecast: &mut Forecast) -> Result<(), Error> {
        for t in self.transforms.iter().rev() {
            t.inverse_transform(&mut forecast.point)?;
            if let Some(intervals) = forecast.intervals.as_mut() {
                t.inverse_transform(&mut intervals.lower)?;
                t.inverse_transform(&mut intervals.upper)?;
            }
        }
        Ok(())
    }
}

/// A transformation that can be applied to a time series.
pub trait Transform: fmt::Debug + Sync + Send {
    /// Fit the transformation to the given time series.
    fn fit(&mut self, data: &[f64]) -> Result<(), Error>;

    /// Apply the transformation to the given time series.
    fn transform(&self, data: &mut [f64]) -> Result<(), Error>;

    /// Apply the inverse transformation to the given time series.
    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error>;

    /// Fit the transformation to the given time series and then apply it.
    fn fit_transform(&mut self, data: &mut [f64]) -> Result<(), Error> {
        self.fit(data)?;
        self.transform(data)?;
        Ok(())
    }

    /// Create a boxed version of the transformation.
    ///
    /// This is useful for creating a `Transform` instance that can be used as
    /// part of a [`Pipeline`].
    fn boxed(self) -> Box<dyn Transform>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}
