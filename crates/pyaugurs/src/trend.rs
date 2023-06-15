//! Bindings for trend models implemented in Python.
//!
//! This module provides the [`PyTrendModel`] struct, which wraps a Python
//! class which implements a trend model. This allows users to implement their
//! trend models in Python and use them in the MSTL algorithm using
//! [`MSTL::custom_trend`][crate::mstl::MSTL::custom_trend].
//!
//! The Python class must implement the following methods:
//!
//! - `fit(self, y: np.ndarray) -> None`
//! - `predict(self, horizon: int, level: float | None = None) -> augurs.Forecast`
//! - `predict_in_sample(self, level: float | None = None) -> augurs.Forecast`
use numpy::ToPyArray;
use pyo3::{exceptions::PyException, prelude::*};

use augurs_mstl::TrendModel;

use crate::Forecast;

/// A Python wrapper for a trend model.
///
/// This allows users to implement their own trend models in Python and use
/// them in the MSTL algorithm using [`MSTL::custom_trend`][crate::mstl::MSTL::custom_trend].
///
/// The Python class must implement the following methods:
///
/// - `fit(self, y: np.ndarray) -> None`
/// - `predict(self, horizon: int, level: float | None = None) -> augurs.Forecast`
/// - `predict_in_sample(self, level: float | None = None) -> augurs.Forecast`
#[pyclass(name = "TrendModel")]
#[derive(Clone, Debug)]
pub struct PyTrendModel {
    model: Py<PyAny>,
}

#[pymethods]
impl PyTrendModel {
    fn __repr__(&self) -> String {
        format!("PyTrendModel(model=\"{}\")", self.name())
    }

    /// Wrap a trend model implemented in Python into a PyTrendModel.
    ///
    /// The returned PyTrendModel can be used in MSTL models using the
    /// `custom_trend` method of the MSTL class.
    #[new]
    pub fn new(model: Py<PyAny>) -> Self {
        Self { model }
    }
}

impl TrendModel for PyTrendModel {
    fn name(&self) -> std::borrow::Cow<'_, str> {
        Python::with_gil(|py| {
            self.model
                .as_ref(py)
                .get_type()
                .name()
                .map(|s| s.to_owned().into())
        })
        .unwrap_or_else(|_| "unknown Python class".into())
    }

    fn fit(&mut self, y: &[f64]) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        Python::with_gil(|py| {
            let np = y.to_pyarray(py);
            self.model.call_method1(py, "fit", (np,))
        })?;
        Ok(())
    }

    fn predict(
        &self,
        horizon: usize,
        level: Option<f64>,
    ) -> Result<augurs_core::Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Python::with_gil(|py| {
            let preds = self
                .model
                .call_method1(py, "predict", (horizon, level))
                .map_err(|e| Box::new(PyException::new_err(format!("error predicting: {e}"))))?;
            let preds: Forecast = preds.extract(py)?;
            Ok(preds.into())
        })
    }

    fn predict_in_sample(
        &self,
        level: Option<f64>,
    ) -> Result<augurs_core::Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Python::with_gil(|py| {
            let preds = self
                .model
                .call_method1(py, "predict_in_sample", (level,))
                .map_err(|e| {
                    Box::new(PyException::new_err(format!(
                        "error predicting in-sample: {e}"
                    )))
                })?;
            let preds: Forecast = preds.extract(py)?;
            Ok(preds.into())
        })
    }
}
