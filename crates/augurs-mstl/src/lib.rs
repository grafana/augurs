#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use stlrs::MstlResult;
use tracing::instrument;

use augurs_core::{Forecast, ForecastIntervals, ModelError, Predict};

// mod approx;
// pub mod mstl;
// mod stationarity;
mod trend;
// mod utils;

pub use crate::trend::{FittedTrendModel, NaiveTrend, TrendModel};

/// Errors that can occur when using this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error occurred while running the MSTL algorithm.
    #[error("fitting MSTL: {0}")]
    MSTL(String),
    /// An error occurred while running the STL algorithm.
    #[error("running STL: {0}")]
    STL(#[from] stlrs::Error),
    /// An error occurred while fitting or predicting using the trend model.
    #[error("trend model error: {0}")]
    TrendModel(Box<dyn std::error::Error + Send + Sync + 'static>),
}

type Result<T> = std::result::Result<T, Error>;

/// A model that uses the [MSTL] to decompose a time series into trend,
/// seasonal and remainder components, and then uses a trend model to
/// forecast the trend component.
///
/// [MSTL]: https://arxiv.org/abs/2107.13462
#[derive(Debug)]
pub struct MSTLModel<T> {
    /// Periodicity of the seasonal components.
    periods: Vec<usize>,
    mstl_params: stlrs::MstlParams,

    trend_model: T,

    impute: bool,
}

impl MSTLModel<NaiveTrend> {
    /// Create a new MSTL model with a naive trend model.
    ///
    /// The naive trend model predicts the last value in the training set
    /// and so is unlikely to be useful for real applications, but it can
    /// be useful for testing, benchmarking and pedagogy.
    pub fn naive(periods: Vec<usize>) -> Self {
        Self::new(periods, NaiveTrend::new())
    }
}

impl<T: TrendModel> MSTLModel<T> {
    /// Return a reference to the trend model.
    pub fn trend_model(&self) -> &T {
        &self.trend_model
    }
}

impl<T: TrendModel> MSTLModel<T> {
    /// Create a new MSTL model with the given trend model.
    pub fn new(periods: Vec<usize>, trend_model: T) -> Self {
        Self {
            periods,
            mstl_params: stlrs::MstlParams::new(),
            trend_model,
            impute: false,
        }
    }

    /// Set whether to impute missing values in the time series.
    ///
    /// If `true`, then missing values will be imputed using
    /// linear interpolation before fitting the model.
    pub fn impute(mut self, impute: bool) -> Self {
        self.impute = impute;
        self
    }

    /// Set the parameters for the MSTL algorithm.
    ///
    /// This can be used to control the parameters for the inner STL algorithm
    /// by using [`stlrs::MstlParams`].
    pub fn mstl_params(mut self, params: stlrs::MstlParams) -> Self {
        self.mstl_params = params;
        self
    }

    /// Fit the model to the given time series.
    ///
    /// # Errors
    ///
    /// If no periods are specified, or if all periods are greater than
    /// half the length of the time series, then an error is returned.
    ///
    /// Any errors returned by the STL algorithm or trend model
    /// are also propagated.
    #[instrument(skip_all)]
    fn fit_impl(&self, y: &[f64]) -> Result<FittedMSTLModel> {
        let y: Vec<f32> = y.iter().copied().map(|y| y as f32).collect::<Vec<_>>();
        let fit = self.mstl_params.fit(&y, &self.periods)?;
        // Determine the differencing term for the trend component.
        let trend = fit.trend();
        let residual = fit.remainder();
        let deseasonalised = trend
            .iter()
            .zip(residual)
            .map(|(t, r)| (t + r) as f64)
            .collect::<Vec<_>>();
        let fitted_trend_model = self
            .trend_model
            .fit(&deseasonalised)
            .map_err(Error::TrendModel)?;
        tracing::trace!(
            trend_model = ?self.trend_model,
            "found best trend model",
        );
        Ok(FittedMSTLModel {
            periods: self.periods.clone(),
            fit,
            fitted_trend_model,
        })
    }
}

/// A model that uses the [MSTL] to decompose a time series into trend,
/// seasonal and remainder components, and then uses a trend model to
/// forecast the trend component.
///
/// [MSTL]: https://arxiv.org/abs/2107.13462
#[derive(Debug)]
pub struct FittedMSTLModel {
    /// Periodicity of the seasonal components.
    periods: Vec<usize>,
    fit: MstlResult,
    fitted_trend_model: Box<dyn FittedTrendModel + Sync + Send>,
}

impl FittedMSTLModel {
    /// Return the MSTL fit of the training data.
    pub fn fit(&self) -> &MstlResult {
        &self.fit
    }
}

impl FittedMSTLModel {
    fn predict_impl(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<()> {
        if horizon == 0 {
            return Ok(());
        }
        self.fitted_trend_model
            .predict_inplace(horizon, level, forecast)
            .map_err(Error::TrendModel)?;
        self.add_seasonal_out_of_sample(forecast);
        Ok(())
    }

    fn predict_in_sample_impl(&self, level: Option<f64>, forecast: &mut Forecast) -> Result<()> {
        self.fitted_trend_model
            .predict_in_sample_inplace(level, forecast)
            .map_err(Error::TrendModel)?;
        self.add_seasonal_in_sample(forecast);
        Ok(())
    }

    fn add_seasonal_in_sample(&self, trend: &mut Forecast) {
        self.fit().seasonal().iter().for_each(|component| {
            let period_contributions = component.iter().zip(trend.point.iter_mut());
            match &mut trend.intervals {
                None => period_contributions.for_each(|(c, p)| *p += *c as f64),
                Some(ForecastIntervals {
                    ref mut lower,
                    ref mut upper,
                    ..
                }) => {
                    period_contributions
                        .zip(lower.iter_mut())
                        .zip(upper.iter_mut())
                        .for_each(|(((c, p), l), u)| {
                            *p += *c as f64;
                            *l += *c as f64;
                            *u += *c as f64;
                        });
                }
            }
        });
    }

    fn add_seasonal_out_of_sample(&self, trend: &mut Forecast) {
        self.periods
            .iter()
            .zip(self.fit().seasonal())
            .for_each(|(period, component)| {
                // For each seasonal period we're going to create a cycle iterator
                // which will repeat the seasonal component every `period` steps.
                // We'll zip it up with the trend point estimates and add the
                // contribution of the seasonal component to the trend.
                // If there are intervals, we'll also add the contribution to those.
                let period_contributions = component
                    .iter()
                    .copied()
                    .skip(component.len() - period)
                    .cycle()
                    .zip(trend.point.iter_mut());
                match &mut trend.intervals {
                    None => period_contributions.for_each(|(c, p)| *p += c as f64),
                    Some(ForecastIntervals {
                        ref mut lower,
                        ref mut upper,
                        ..
                    }) => {
                        period_contributions
                            .zip(lower.iter_mut())
                            .zip(upper.iter_mut())
                            .for_each(|(((c, p), l), u)| {
                                *p += c as f64;
                                *l += c as f64;
                                *u += c as f64;
                            });
                    }
                }
            });
    }
}

impl ModelError for Error {}

impl<T: TrendModel> augurs_core::Fit for MSTLModel<T> {
    type Fitted = FittedMSTLModel;
    type Error = Error;
    fn fit(&self, y: &[f64]) -> Result<Self::Fitted> {
        self.fit_impl(y)
    }
}

impl Predict for FittedMSTLModel {
    type Error = Error;

    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<()> {
        self.predict_impl(horizon, level, forecast)
    }

    fn predict_in_sample_inplace(&self, level: Option<f64>, forecast: &mut Forecast) -> Result<()> {
        self.predict_in_sample_impl(level, forecast)
    }

    fn training_data_size(&self) -> usize {
        self.fit().trend().len()
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use augurs_core::prelude::*;
    use augurs_testing::data::VIC_ELEC;

    use crate::{trend::NaiveTrend, ForecastIntervals, MSTLModel};

    #[track_caller]
    fn assert_all_close(actual: &[f64], expected: &[f64]) {
        for (actual, expected) in actual.iter().zip(expected) {
            if actual.is_nan() {
                assert!(expected.is_nan());
            } else {
                assert_approx_eq!(actual, expected, 1e-1);
            }
        }
    }

    #[test]
    fn results_match_r() {
        let y = VIC_ELEC.clone();

        let mut stl_params = stlrs::params();
        stl_params
            .seasonal_degree(0)
            .seasonal_jump(1)
            .trend_degree(1)
            .trend_jump(1)
            .low_pass_degree(1)
            .inner_loops(2)
            .outer_loops(0);
        let mut mstl_params = stlrs::MstlParams::new();
        mstl_params.stl_params(stl_params);
        let periods = vec![24, 24 * 7];
        let trend_model = NaiveTrend::new();
        let mstl = MSTLModel::new(periods, trend_model).mstl_params(mstl_params);
        let fit = mstl.fit(&y).unwrap();

        let in_sample = fit.predict_in_sample(0.95).unwrap();
        let expected_in_sample = vec![
            f64::NAN,
            7952.216,
            7269.439,
            6878.110,
            6606.999,
            6402.581,
            6659.523,
            7457.488,
            8111.359,
            8693.762,
            9255.807,
            9870.213,
        ];
        assert_eq!(in_sample.point.len(), y.len());
        assert_all_close(&in_sample.point, &expected_in_sample);

        let out_of_sample = fit.predict(10, 0.95).unwrap();
        let expected_out_of_sample: Vec<f64> = vec![
            8920.670, 8874.234, 8215.508, 7782.726, 7697.259, 8216.241, 9664.907, 10914.452,
            11536.929, 11664.737,
        ];
        let expected_out_of_sample_lower = vec![
            8700.984, 8563.551, 7835.001, 7343.354, 7206.026, 7678.122, 9083.672, 10293.087,
            10877.871, 10970.029,
        ];
        let expected_out_of_sample_upper = vec![
            9140.356, 9184.917, 8596.016, 8222.098, 8188.491, 8754.359, 10246.141, 11535.818,
            12195.987, 12359.445,
        ];
        assert_eq!(out_of_sample.point.len(), 10);
        assert_all_close(&out_of_sample.point, &expected_out_of_sample);
        let ForecastIntervals { lower, upper, .. } = out_of_sample.intervals.unwrap();
        assert_eq!(lower.len(), 10);
        assert_eq!(upper.len(), 10);
        assert_all_close(&lower, &expected_out_of_sample_lower);
        assert_all_close(&upper, &expected_out_of_sample_upper);
    }

    #[test]
    fn predict_zero_horizon() {
        let y = VIC_ELEC.clone();

        let mut stl_params = stlrs::params();
        stl_params
            .seasonal_degree(0)
            .seasonal_jump(1)
            .trend_degree(1)
            .trend_jump(1)
            .low_pass_degree(1)
            .inner_loops(2)
            .outer_loops(0);
        let mut mstl_params = stlrs::MstlParams::new();
        mstl_params.stl_params(stl_params);
        let periods = vec![24, 24 * 7];
        let trend_model = NaiveTrend::new();
        let mstl = MSTLModel::new(periods, trend_model).mstl_params(mstl_params);
        let fit = mstl.fit(&y).unwrap();
        let forecast = fit.predict(0, 0.95).unwrap();
        assert!(forecast.point.is_empty());
        let ForecastIntervals { lower, upper, .. } = forecast.intervals.unwrap();
        assert!(lower.is_empty());
        assert!(upper.is_empty());
    }
}
