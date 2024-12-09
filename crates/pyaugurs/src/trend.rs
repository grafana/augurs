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
use pyo3::{exceptions::PyException, prelude::*, types::PyAnyMethods};

use augurs_mstl::{FittedTrendModel, TrendModel};

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
#[derive(Debug)]
pub(crate) struct PyTrendModel {
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
    pub(crate) fn new(model: Py<PyAny>) -> Self {
        Self { model }
    }
}

impl TrendModel for PyTrendModel {
    fn name(&self) -> std::borrow::Cow<'_, str> {
        Python::with_gil(|py| {
            self.model
                .bind(py)
                .get_type()
                .name()
                .map(|s| s.to_string().into())
        })
        .unwrap_or_else(|_| "unknown Python class".into())
    }

    fn fit(
        &self,
        y: &mut [f64],
    ) -> Result<
        Box<dyn FittedTrendModel + Sync + Send>,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    > {
        let model = Python::with_gil(|py| {
            let np = y.to_pyarray(py);
            self.model.call_method1(py, "fit", (np,))?;
            Ok::<_, PyErr>(self.model.clone_ref(py))
        })?;
        Ok(Box::new(PyFittedTrendModel { model }) as _)
    }
}

/// A wrapper for a Python trend model that has been fitted to data.
#[derive(Debug)]
pub(crate) struct PyFittedTrendModel {
    model: Py<PyAny>,
}

impl FittedTrendModel for PyFittedTrendModel {
    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut augurs_core::Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        Python::with_gil(|py| {
            let preds = self
                .model
                .call_method1(py, "predict", (horizon, level))
                .map_err(|e| Box::new(PyException::new_err(format!("error predicting: {e}"))))?;
            let preds: Forecast = preds.extract(py)?;
            *forecast = preds.into();
            Ok(())
        })
    }

    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut augurs_core::Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
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
            *forecast = preds.into();
            Ok(())
        })
    }

    fn training_data_size(&self) -> Option<usize> {
        None
    }
}
