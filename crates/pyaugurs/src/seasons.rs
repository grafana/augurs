//! Bindings for seasonality detection.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use augurs_seasons::{Detector, PeriodogramDetector};

/// Detect the seasonal periods in a time series.
#[pyfunction]
pub fn seasonalities(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    min_period: Option<u32>,
    max_period: Option<u32>,
    threshold: Option<f64>,
) -> PyResult<Py<PyArray1<u32>>> {
    let mut builder = PeriodogramDetector::builder();

    if let Some(min_period) = min_period {
        builder = builder.min_period(min_period);
    }
    if let Some(max_period) = max_period {
        builder = builder.max_period(max_period);
    }
    if let Some(threshold) = threshold {
        builder = builder.threshold(threshold);
    }

    Ok(builder.build().detect(y.as_slice()?).to_pyarray(py).into())
}
