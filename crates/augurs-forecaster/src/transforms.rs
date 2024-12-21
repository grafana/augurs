/*!
Data transformations.

This module contains the `Transform` enum, which contains various
predefined transformations. The enum contains various methods for
creating new instances of the various transformations, as well as
the `transform` and `inverse_transform` methods, which allow you to
apply a transformation to a time series and its inverse, respectively.
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

/// Transforms and Transform implementations.
///
/// The `Transforms` struct is a collection of `Transform` instances that can be applied to a time series.
/// The `Transform` enum represents a single transformation that can be applied to a time series.
#[derive(Debug, Default)]
pub struct Pipeline(Vec<Box<dyn Transform>>);

impl Pipeline {
    /// Create a new `Pipeline` with the given transforms.
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self(transforms)
    }

    /// Apply the transformations to the given time series.
    pub fn transform(&mut self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.0.iter_mut() {
            t.transform(input)?;
        }
        Ok(())
    }

    /// Apply the inverse transformations to the given time series.
    pub fn inverse_transform(&self, input: &mut [f64]) -> Result<(), Error> {
        for t in self.0.iter().rev() {
            t.inverse_transform(input)?;
        }
        Ok(())
    }

    /// Apply the inverse transformations to the given forecast.
    pub(crate) fn inverse_transform_forecast(&self, forecast: &mut Forecast) -> Result<(), Error> {
        for t in self.0.iter().rev() {
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
    /// Apply the transformation to the given time series.
    fn transform(&mut self, data: &mut [f64]) -> Result<(), Error>;

    /// Apply the inverse transformation to the given time series.
    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error>;

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
