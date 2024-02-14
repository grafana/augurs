//! Python bindings for the augurs time series framework.
//!
//! These bindings are intended to be used from Python, and are not useful from Rust.
//! The documentation here is useful for understanding the Python API, however.
//!
//! See the crate README for information on Python API usage and installation.
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

pub mod ets;
pub mod mstl;
pub mod seasons;
pub mod trend;

/// Forecasts produced by augurs models.
#[derive(Debug, Clone)]
#[pyclass]
pub struct Forecast {
    inner: augurs_core::Forecast,
}

impl From<augurs_core::Forecast> for Forecast {
    fn from(inner: augurs_core::Forecast) -> Self {
        Self { inner }
    }
}

impl From<Forecast> for augurs_core::Forecast {
    fn from(forecast: Forecast) -> Self {
        forecast.inner
    }
}

#[pymethods]
impl Forecast {
    #[new]
    fn new(
        py: Python<'_>,
        point: Py<PyArray1<f64>>,
        level: Option<f64>,
        lower: Option<Py<PyArray1<f64>>>,
        upper: Option<Py<PyArray1<f64>>>,
    ) -> pyo3::PyResult<Self> {
        Ok(Self {
            inner: augurs_core::Forecast {
                point: point.extract(py)?,
                intervals: level
                    .zip(lower)
                    .zip(upper)
                    .map(|((level, lower), upper)| {
                        Ok::<_, PyErr>(augurs_core::ForecastIntervals {
                            level,
                            lower: lower.extract(py)?,
                            upper: upper.extract(py)?,
                        })
                    })
                    .transpose()?,
            },
        })
    }

    fn __repr__(&self) -> String {
        let intervals = self.inner.intervals.as_ref();
        format!(
            "Forecast(point={:?}, level={:?}, lower={:?}, upper={:?})",
            self.inner.point,
            intervals.map(|x| x.level),
            intervals.map(|x| &x.lower),
            intervals.map(|x| &x.upper)
        )
    }

    /// Get the point forecast.
    fn point(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        // Use `to_pyarray` to allocate a new array on the Python heap.
        // We could also use `into_pyarray` to construct the
        // numpy arrays in the Rust heap; let's see which ends up being
        // faster and more convenient.
        self.inner.point.to_pyarray(py).into()
    }

    /// Get the lower prediction interval.
    fn lower(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.inner
            .intervals
            .as_ref()
            .map(|x| x.lower.to_pyarray(py).into())
    }

    /// Get the upper prediction interval.
    fn upper(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.inner
            .intervals
            .as_ref()
            .map(|x| x.upper.to_pyarray(py).into())
    }
}

/// Python bindings for the augurs time series framework.
#[pymodule]
fn augurs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<ets::AutoETS>()?;
    m.add_class::<mstl::MSTL>()?;
    m.add_class::<trend::PyTrendModel>()?;
    m.add_class::<Forecast>()?;
    m.add_function(wrap_pyfunction!(seasons::seasonalities, m)?)?;
    Ok(())
}
