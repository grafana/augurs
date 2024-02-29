use crate::{Forecast, ModelError};

/// A new, unfitted time series forecasting model.
pub trait Fit {
    /// The type of the fitted model produced by the `fit` method.
    type Fitted: Predict;

    /// The type of error returned when fitting the model.
    type Error: ModelError;

    /// Fit the model to the training data.
    fn fit(&self, y: &[f64]) -> Result<Self::Fitted, Self::Error>;
}

impl<F> Fit for Box<F>
where
    F: Fit,
{
    type Fitted = F::Fitted;
    type Error = F::Error;
    fn fit(&self, y: &[f64]) -> Result<Self::Fitted, Self::Error> {
        (**self).fit(y)
    }
}

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
