//! Trend models.
//!
//! Contains the [`TrendModel`] trait and an implementation of a basic
//! naive trend model.
// TODO: decide where this should live. Perhaps it's more general than just MSTL?

use std::{
    borrow::Cow,
    fmt::{self, Debug},
};

use crate::{Forecast, ForecastIntervals};

/// A trend model.
///
/// Trend models are used to model the trend component of a time series.
/// Examples implemented in other languages include ARIMA, Theta and ETS.
///
/// You can implement this trait for your own trend models.
pub trait TrendModel: Debug {
    /// Return the name of the trend model.
    fn name(&self) -> Cow<'_, str>;

    /// Fit the model to the given time series.
    ///
    /// This method is called once before any calls to `predict` or `predict_in_sample`.
    ///
    /// Implementations should store any state required for prediction in the struct itself.
    fn fit(&mut self, y: &[f64]) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>;

    /// Produce a forecast for the next `horizon` time points.
    ///
    /// The `level` parameter specifies the confidence level for the prediction intervals.
    /// Where possible, implementations should provide prediction intervals
    /// alongside the point forecasts if `level` is not `None`.
    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>;

    /// Produce in-sample predictions.
    ///
    /// In-sample predictions are used to assess the fit of the model to the training data.
    ///
    /// The `level` parameter specifies the confidence level for the prediction intervals.
    /// Where possible, implementations should provide prediction intervals
    /// alongside the point forecasts if `level` is not `None`.
    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>;

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
        level: Option<f64>,
    ) -> Result<Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
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
    fn predict_in_sample(
        &self,
        level: Option<f64>,
    ) -> Result<Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut forecast = level
            .zip(self.training_data_size())
            .map(|(l, c)| Forecast::with_capacity_and_level(c, l))
            .unwrap_or_else(|| Forecast::with_capacity(0));
        self.predict_in_sample_inplace(level, &mut forecast)?;
        Ok(forecast)
    }

    /// Return the number of training data points used to fit the model.
    fn training_data_size(&self) -> Option<usize>;
}

impl<T: TrendModel + ?Sized> TrendModel for Box<T> {
    fn name(&self) -> Cow<'_, str> {
        (**self).name()
    }

    fn fit(&mut self, y: &[f64]) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        (**self).fit(y)
    }

    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        (**self).predict_inplace(horizon, level, forecast)
    }

    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        (**self).predict_in_sample_inplace(level, forecast)
    }

    fn training_data_size(&self) -> Option<usize> {
        (**self).training_data_size()
    }
}

/// A naive trend model that predicts the last value in the training set
/// for all future time points.
#[derive(Clone, Default)]
pub struct NaiveTrend {
    fitted: Option<Vec<f64>>,
    last_value: Option<f64>,
    sigma_squared: Option<f64>,
}

impl fmt::Debug for NaiveTrend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NaiveTrend")
            .field(
                "y",
                &self
                    .fitted
                    .as_ref()
                    .map(|y| format!("<omitted vec, length {}>", y.len())),
            )
            .field("last_value", &self.last_value)
            .field("sigma", &self.sigma_squared)
            .finish()
    }
}

impl NaiveTrend {
    /// Create a new naive trend model.
    pub const fn new() -> Self {
        Self {
            fitted: None,
            last_value: None,
            sigma_squared: None,
        }
    }

    fn prediction_intervals(
        &self,
        preds: impl Iterator<Item = f64>,
        level: f64,
        sigma: impl Iterator<Item = f64>,
        intervals: &mut ForecastIntervals,
    ) {
        intervals.level = level;
        let z = distrs::Normal::ppf(0.5 + level / 2.0, 0.0, 1.0);
        (intervals.lower, intervals.upper) = preds
            .zip(sigma)
            .map(|(p, s)| (p - z * s, p + z * s))
            .unzip();
    }
}

impl TrendModel for NaiveTrend {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Naive")
    }

    fn fit(&mut self, y: &[f64]) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.last_value = Some(y[y.len() - 1]);
        let fitted: Vec<f64> = std::iter::once(f64::NAN)
            .chain(y.iter().copied())
            .take(y.len())
            .collect();
        let sigma_squared = y
            .iter()
            .zip(&fitted)
            .filter_map(|(y, f)| {
                if f.is_nan() {
                    None
                } else {
                    Some((y - f).powi(2))
                }
            })
            .sum::<f64>()
            / (y.len() - 1) as f64;
        self.fitted = Some(fitted);
        self.sigma_squared = Some(sigma_squared);
        Ok(())
    }

    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        match self.last_value.zip(self.sigma_squared) {
            Some((l, sigma)) => {
                forecast.point = vec![l; horizon];
                if let Some(level) = level {
                    let sigmas = (1..horizon + 1).map(|step| ((step as f64) * sigma).sqrt());
                    let intervals = forecast
                        .intervals
                        .get_or_insert_with(|| ForecastIntervals::with_capacity(level, horizon));
                    self.prediction_intervals(std::iter::repeat(l), level, sigmas, intervals);
                }
                Ok(())
            }
            None => Err("model not fit")?,
        }
    }

    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        Ok(self
            .fitted
            .as_ref()
            .zip(self.sigma_squared)
            .map(|(fitted, sigma)| {
                forecast.point = fitted.clone();
                if let Some(level) = level {
                    let intervals = forecast.intervals.get_or_insert_with(|| {
                        ForecastIntervals::with_capacity(level, fitted.len())
                    });
                    self.prediction_intervals(
                        fitted.iter().copied(),
                        level,
                        std::iter::repeat(sigma.sqrt()),
                        intervals,
                    );
                }
            })
            .ok_or("model not fit")?)
    }

    fn training_data_size(&self) -> Option<usize> {
        self.fitted.as_ref().map(Vec::len)
    }
}
