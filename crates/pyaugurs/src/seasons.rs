//! Bindings for seasonality detection.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use augurs_seasons::{Detector, PeriodogramDetector};

/// Detect the seasonal periods in a time series.
#[pyfunction]
pub fn seasonalities(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyArray1<usize>>> {
    Ok(PeriodogramDetector::builder()
        .build(y.as_slice()?)
        .detect()
        .collect::<Vec<_>>()
        .to_pyarray(py)
        .into())
}
