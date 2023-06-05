//! Bindings for Multiple Seasonal Trend using LOESS (MSTL).
use std::borrow::Cow;

use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyException, prelude::*, types::PyType};

use augurs_ets::AutoETS;
use augurs_mstl::{Fit, MSTLModel, TrendModel, Unfit};

use crate::{trend::PyTrendModel, Forecast};

#[derive(Debug)]
enum MSTLEnum<T> {
    Unfit(MSTLModel<T, Unfit>),
    Fit(MSTLModel<T, Fit>),
}

/// A MSTL model.
#[derive(Debug)]
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct MSTL {
    inner: Option<MSTLEnum<Box<dyn TrendModel + Sync + Send>>>,
}

#[pymethods]
impl MSTL {
    fn __repr__(&self) -> String {
        format!(
            "MSTL(fit_state=\"{}\", trend_model=\"{}\")",
            match &self.inner {
                Some(MSTLEnum::Unfit(_)) => "unfit",
                Some(MSTLEnum::Fit(_)) => "fit",
                None => "unknown",
            },
            match &self.inner {
                Some(MSTLEnum::Unfit(x)) => x.trend_model().name(),
                Some(MSTLEnum::Fit(x)) => x.trend_model().name(),
                None => Cow::Borrowed("unknown"),
            }
        )
    }

    /// Create a new MSTL model with the given periods using the `AutoETS` trend model.
    #[classmethod]
    pub fn ets(_cls: &PyType, periods: Vec<usize>) -> Self {
        let ets = AutoETS::non_seasonal();
        Self {
            inner: Some(MSTLEnum::Unfit(MSTLModel::new(periods, Box::new(ets)))),
        }
    }

    /// Create a new MSTL model with the given periods using provided trend model.
    #[classmethod]
    pub fn custom_trend(_cls: &PyType, periods: Vec<usize>, trend_model: PyTrendModel) -> Self {
        Self {
            inner: Some(MSTLEnum::Unfit(MSTLModel::new(
                periods,
                Box::new(trend_model),
            ))),
        }
    }

    /// Fit the model to the given time series.
    pub fn fit(&mut self, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.inner = match std::mem::take(&mut self.inner) {
            Some(MSTLEnum::Unfit(inner)) => {
                Some(MSTLEnum::Fit(inner.fit(y.as_slice()?).map_err(|e| {
                    PyException::new_err(format!("error fitting model: {e}"))
                })?))
            }
            x => x,
        };
        Ok(())
    }

    /// Predict the next `horizon` values, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    pub fn predict(&self, horizon: usize, level: Option<f64>) -> PyResult<Forecast> {
        match &self.inner {
            Some(MSTLEnum::Fit(inner)) => inner
                .predict(horizon, level)
                .map(Forecast::from)
                .map_err(|e| PyException::new_err(format!("error predicting: {e}"))),
            _ => Err(PyException::new_err("model not fit yet")),
        }
    }

    /// Produce in-sample forecasts, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    pub fn predict_in_sample(&self, level: Option<f64>) -> PyResult<Forecast> {
        match &self.inner {
            Some(MSTLEnum::Fit(inner)) => inner
                .predict_in_sample(level)
                .map(Forecast::from)
                .map_err(|e| PyException::new_err(format!("error predicting: {e}"))),
            _ => Err(PyException::new_err("model not fit yet")),
        }
    }
}
