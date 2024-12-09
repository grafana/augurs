//! Bindings for Multiple Seasonal Trend using LOESS (MSTL).

use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyException, prelude::*, types::PyType};

use augurs_ets::{trend::AutoETSTrendModel, AutoETS};
use augurs_forecaster::Forecaster;
use augurs_mstl::{MSTLModel, TrendModel};

use crate::{trend::PyTrendModel, Forecast};

/// A MSTL model.
#[derive(Debug)]
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct MSTL {
    forecaster: Forecaster<MSTLModel<Box<dyn TrendModel + Sync + Send>>>,
    trend_model_name: String,
    fit: bool,
}

#[pymethods]
impl MSTL {
    fn __repr__(&self) -> String {
        format!(
            "MSTL(fit=\"{}\", trend_model=\"{}\")",
            match self.fit {
                false => "unfit",
                true => "fit",
            },
            &self.trend_model_name,
        )
    }

    /// Create a new MSTL model with the given periods using the `AutoETS` trend model.
    #[classmethod]
    pub fn ets(_cls: &Bound<'_, PyType>, periods: Vec<usize>) -> Self {
        let ets = AutoETSTrendModel::from(AutoETS::non_seasonal());
        let trend_model_name = ets.name().to_string();
        Self {
            forecaster: Forecaster::new(MSTLModel::new(periods, Box::new(ets))),
            trend_model_name,
            fit: false,
        }
    }

    /// Create a new MSTL model with the given periods using the custom Python trend model.
    ///
    /// The custom trend model must implement the following methods:
    ///
    /// - `fit(self, y: np.ndarray) -> None`
    /// - `predict(self, horizon: int, level: float | None = None) -> augurs.Forecast`
    /// - `predict_in_sample(self, level: float | None = None) -> augurs.Forecast`
    #[classmethod]
    pub fn custom_trend(
        _cls: &Bound<'_, PyType>,
        periods: Vec<usize>,
        trend_model: Py<PyAny>,
    ) -> Self {
        let trend_model_name = Python::with_gil(|py| {
            let trend_model = trend_model.bind(py).get_type();
            trend_model
                .name()
                .map_or_else(|_| "unknown Python class".into(), |s| s.to_string())
        });
        Self {
            forecaster: Forecaster::new(MSTLModel::new(
                periods,
                Box::new(PyTrendModel::new(trend_model)),
            )),
            trend_model_name,
            fit: false,
        }
    }

    /// Fit the model to the given time series.
    pub fn fit(&mut self, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        self.forecaster
            .fit(y.as_slice()?)
            .map_err(|e| PyException::new_err(format!("error fitting model: {e}")))?;
        self.fit = true;
        Ok(())
    }

    /// Predict the next `horizon` values, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    #[pyo3(signature = (horizon, level=None))]
    pub fn predict(&self, horizon: usize, level: Option<f64>) -> PyResult<Forecast> {
        self.forecaster
            .predict(horizon, level)
            .map(Forecast::from)
            .map_err(|e| PyException::new_err(format!("error predicting: {e}")))
    }

    /// Produce in-sample forecasts, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    #[pyo3(signature = (level=None))]
    pub fn predict_in_sample(&self, level: Option<f64>) -> PyResult<Forecast> {
        self.forecaster
            .predict_in_sample(level)
            .map(Forecast::from)
            .map_err(|e| PyException::new_err(format!("error predicting: {e}")))
    }
}
