//! Bindings for AutoETS model search.
use numpy::PyReadonlyArrayDyn;
use pyo3::{exceptions::PyException, prelude::*};

use crate::Forecast;

/// Automatic exponential smoothing model search.
#[derive(Debug)]
#[pyclass]
pub struct AutoETS {
    inner: augurs_ets::AutoETS,
}

#[pymethods]
impl AutoETS {
    /// Create a new `AutoETS` model search instance.
    ///
    /// # Errors
    ///
    /// If the `spec` string is invalid, this function returns an error.
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(season_length: usize, spec: String) -> PyResult<Self> {
        let inner = augurs_ets::AutoETS::new(season_length, spec.as_str())
            .map_err(|e| PyException::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "AutoETS(spec=\"{}\", season_length={})",
            self.inner.spec(),
            self.inner.season_length()
        )
    }

    /// Search for the best model, fitting it to the data.
    ///
    /// The model will be stored on the inner `AutoETS` instance, after which
    /// forecasts can be produced using its `predict` method.
    ///
    /// # Errors
    ///
    /// If no model can be found, or if any parameters are invalid, this function
    /// returns an error.
    pub fn fit(&mut self, y: PyReadonlyArrayDyn<'_, f64>) -> PyResult<()> {
        self.inner
            .fit(y.as_slice()?)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        Ok(())
    }

    /// Predict the next `horizon` values using the best model, optionally including
    /// prediction intervals at the specified level.
    ///
    /// `level` should be a float between 0 and 1 representing the confidence level.
    ///
    /// # Errors
    ///
    /// This function will return an error if no model has been fit yet (using [`AutoETS::fit`]).
    pub fn predict(&self, horizon: usize, level: Option<f64>) -> PyResult<Forecast> {
        self.inner
            .predict(horizon, level)
            .map(Forecast::from)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Get the in-sample predictions for the model, optionally including
    /// prediction intervals at the specified level.
    ///
    /// `level` should be a float between 0 and 1 representing the confidence level.
    ///
    /// # Errors
    ///
    /// This function will return an error if no model has been fit yet (using [`AutoETS::fit`]).
    pub fn predict_in_sample(&self, level: Option<f64>) -> PyResult<Forecast> {
        self.inner
            .predict_in_sample(level)
            .map(Forecast::from)
            .map_err(|e| PyException::new_err(e.to_string()))
    }
}
