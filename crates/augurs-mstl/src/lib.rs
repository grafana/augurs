#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use std::marker::PhantomData;

use stlrs::MstlResult;
use tracing::instrument;

use augurs_core::{Forecast, ForecastIntervals};

// mod approx;
// pub mod mstl;
// mod stationarity;
mod trend;
// mod utils;

pub use crate::trend::{NaiveTrend, TrendModel};

/// A marker struct indicating that a model is fit.
#[derive(Debug, Clone, Copy)]
pub struct Fit;

/// A marker struct indicating that a model is unfit.
#[derive(Debug, Clone, Copy)]
pub struct Unfit;

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
pub struct MSTLModel<T, F> {
    /// Periodicity of the seasonal components.
    periods: Vec<usize>,
    mstl_params: stlrs::MstlParams,

    state: PhantomData<F>,

    fit: Option<MstlResult>,
    trend_model: T,
}

impl MSTLModel<NaiveTrend, Unfit> {
    /// Create a new MSTL model with a naive trend model.
    ///
    /// The naive trend model predicts the last value in the training set
    /// and so is unlikely to be useful for real applications, but it can
    /// be useful for testing, benchmarking and pedagogy.
    pub fn naive(periods: Vec<usize>) -> Self {
        Self::new(periods, NaiveTrend::new())
    }
}

impl<T: TrendModel, F> MSTLModel<T, F> {
    /// Return a reference to the trend model.
    pub fn trend_model(&self) -> &T {
        &self.trend_model
    }
}

impl<T: TrendModel> MSTLModel<T, Unfit> {
    /// Create a new MSTL model with the given trend model.
    pub fn new(periods: Vec<usize>, trend_model: T) -> Self {
        Self {
            periods,
            state: PhantomData,
            mstl_params: stlrs::MstlParams::new(),
            fit: None,
            trend_model,
        }
    }

    /// Set the parameters for the MSTL algorithm.
    ///
    /// This can be used to control the parameters for the inner STL algorithm
    /// by using [`MstlParams::stl_params`].
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
    pub fn fit(mut self, y: &[f64]) -> Result<MSTLModel<T, Fit>> {
        let y = y.iter().copied().map(|y| y as f32).collect::<Vec<_>>();
        let fit = self.mstl_params.fit(&y, &self.periods)?;
        // Determine the differencing term for the trend component.
        let trend = fit.trend();
        let residual = fit.remainder();
        let deseasonalised = trend
            .iter()
            .zip(residual)
            .map(|(t, r)| (t + r) as f64)
            .collect::<Vec<_>>();
        self.trend_model
            .fit(&deseasonalised)
            .map_err(Error::TrendModel)?;
        tracing::trace!(
            trend_model = ?self.trend_model,
            "found best trend model",
        );
        Ok(MSTLModel {
            periods: self.periods,
            mstl_params: self.mstl_params,
            state: PhantomData,
            fit: Some(fit),
            trend_model: self.trend_model,
        })
    }
}

impl<T: TrendModel> MSTLModel<T, Fit> {
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
    pub fn predict(&self, horizon: usize, level: impl Into<Option<f64>>) -> Result<Forecast> {
        self.predict_impl(horizon, level.into())
    }

    fn predict_impl(&self, horizon: usize, level: Option<f64>) -> Result<Forecast> {
        if horizon == 0 {
            return Ok(Forecast {
                point: vec![],
                intervals: level.map(ForecastIntervals::empty),
            });
        }
        let mut out_of_sample = self
            .trend_model
            .predict(horizon, level)
            .map_err(Error::TrendModel)?;
        self.add_seasonal_out_of_sample(&mut out_of_sample);
        Ok(out_of_sample)
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
    pub fn predict_in_sample(&self, level: impl Into<Option<f64>>) -> Result<Forecast> {
        self.predict_in_sample_impl(level.into())
    }

    fn predict_in_sample_impl(&self, level: Option<f64>) -> Result<Forecast> {
        let mut in_sample = self
            .trend_model
            .predict_in_sample(level)
            .map_err(Error::TrendModel)?;
        self.add_seasonal_in_sample(&mut in_sample);
        Ok(in_sample)
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

    /// Return the MSTL fit of the training data.
    pub fn fit(&self) -> &MstlResult {
        self.fit.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

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
