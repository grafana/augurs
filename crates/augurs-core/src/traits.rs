use crate::{Forecast, ModelError};

/// Trait for data that can be used as an input to [`Fit`].
///
/// This trait is implemented for a number of types including slices, arrays, and
/// vectors. It is also implemented for references to these types.
pub trait Data {
    /// Return the data as a slice of `f64`.
    fn as_slice(&self) -> &[f64];
}

impl<const N: usize> Data for [f64; N] {
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl Data for &[f64] {
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl Data for &mut [f64] {
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl Data for Vec<f64> {
    fn as_slice(&self) -> &[f64] {
        self.as_slice()
    }
}

impl<T> Data for &T
where
    T: Data,
{
    fn as_slice(&self) -> &[f64] {
        (**self).as_slice()
    }
}

impl<T> Data for &mut T
where
    T: Data,
{
    fn as_slice(&self) -> &[f64] {
        (**self).as_slice()
    }
}

/// Trait for data that can be used in the forecaster.
///
/// This trait is implemented for a number of types including slices, arrays, and
/// vectors. It is also implemented for references to these types.
pub trait MutableData: Data {
    /// Update the `y` values to those in the provided slice.
    fn set(&mut self, y: Vec<f64>);
}

impl<const N: usize> MutableData for [f64; N] {
    fn set(&mut self, y: Vec<f64>) {
        self.copy_from_slice(y.as_slice());
    }
}

impl MutableData for &mut [f64] {
    fn set(&mut self, y: Vec<f64>) {
        self.copy_from_slice(y.as_slice());
    }
}

impl MutableData for Vec<f64> {
    fn set(&mut self, y: Vec<f64>) {
        self.copy_from_slice(y.as_slice());
    }
}

impl<T> MutableData for &mut T
where
    T: MutableData,
{
    fn set(&mut self, y: Vec<f64>) {
        (**self).set(y);
    }
}

/// A new, unfitted time series forecasting model.
pub trait Fit {
    /// The type of the training data used to fit the model.
    type TrainingData<'a>: Data
    where
        Self: 'a;

    /// The type of the fitted model produced by the `fit` method.
    type Fitted: Predict;

    /// The type of error returned when fitting the model.
    type Error: ModelError;

    /// Fit the model to the training data.
    fn fit<'a, 'b: 'a>(&'b self, y: Self::TrainingData<'a>) -> Result<Self::Fitted, Self::Error>;
}

// impl<'a, F, TD> Fit for Box<F>
// where
//     F: Fit<TrainingData<'a> = TD>,
//     TD: Data,
// {
//     type TrainingData = TD;
//     type Fitted = F::Fitted;
//     type Error = F::Error;
//     fn fit(&self, y: Self::TrainingData<'a>) -> Result<Self::Fitted, Self::Error> {
//         (**self).fit(y)
//     }
// }

/// A fitted time series forecasting model.
pub trait Predict {
    /// The type of error returned when predicting with the model.
    type Error: ModelError;

    /// Calculate the in-sample predictions, storing the results in the provided
    /// [`Forecast`] struct.
    ///
    /// The predictions are point forecasts and optionally include
    /// prediction intervals at the specified `level`.
    ///
    /// `level` should be a float between 0 and 1 representing the
    /// confidence level of the prediction intervals. If `None` then
    /// no prediction intervals are returned.
    ///
    /// # Errors
    ///
    /// Any errors returned by the trend model are propagated.
    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Self::Error>;

    /// Calculate the n-ahead predictions for the given horizon, storing the results in the
    /// provided [`Forecast`] struct.
    ///
    /// The predictions are point forecasts and optionally include
    /// prediction intervals at the specified `level`.
    ///
    /// `level` should be a float between 0 and 1 representing the
    /// confidence level of the prediction intervals. If `None` then
    /// no prediction intervals are returned.
    ///
    /// # Errors
    ///
    /// Any errors returned by the trend model are propagated.
    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Self::Error>;

    /// Return the number of training data points used to fit the model.
    ///
    /// This is used for pre-allocating the in-sample forecasts.
    fn training_data_size(&self) -> usize;

    /// Return the n-ahead predictions for the given horizon.
    ///
    /// The predictions are point forecasts and optionally include
    /// prediction intervals at the specified `level`.
    ///
    /// `level` should be a float between 0 and 1 representing the
    /// confidence level of the prediction intervals. If `None` then
    /// no prediction intervals are returned.
    ///
    /// # Errors
    ///
    /// Any errors returned by the trend model are propagated.
    fn predict(
        &self,
        horizon: usize,
        level: impl Into<Option<f64>>,
    ) -> Result<Forecast, Self::Error> {
        let level = level.into();
        let mut forecast = level
            .map(|l| Forecast::with_capacity_and_level(horizon, l))
            .unwrap_or_else(|| Forecast::with_capacity(horizon));
        self.predict_inplace(horizon, level, &mut forecast)?;
        Ok(forecast)
    }

    /// Return the in-sample predictions.
    ///
    /// The predictions are point forecasts and optionally include
    /// prediction intervals at the specified `level`.
    ///
    /// `level` should be a float between 0 and 1 representing the
    /// confidence level of the prediction intervals. If `None` then
    /// no prediction intervals are returned.
    ///
    /// # Errors
    ///
    /// Any errors returned by the trend model are propagated.
    fn predict_in_sample(&self, level: impl Into<Option<f64>>) -> Result<Forecast, Self::Error> {
        let level = level.into();
        let mut forecast = level
            .map(|l| Forecast::with_capacity_and_level(self.training_data_size(), l))
            .unwrap_or_else(|| Forecast::with_capacity(self.training_data_size()));
        self.predict_in_sample_inplace(level, &mut forecast)?;
        Ok(forecast)
    }
}
