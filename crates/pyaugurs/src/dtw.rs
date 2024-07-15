//! Bindings for Dynamic Time Warping (DTW).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use augurs_dtw::Euclidean;

#[derive(Debug)]
#[pyclass]
pub struct Dtw {
    inner: augurs_dtw::Dtw<Euclidean>,
}

#[pymethods]
impl Dtw {
    fn __repr__(&self) -> String {
        format!(
            "Dtw(window={})",
            match self.inner.window() {
                Some(window) => window.to_string(),
                None => "None".to_string(),
            },
        )
    }

    #[new]
    fn new(window: Option<usize>) -> Self {
        let mut inner = augurs_dtw::Dtw::euclidean();
        if let Some(window) = window {
            inner = inner.with_window(window);
        }
        Self { inner }
    }

    pub fn distance(
        &self,
        a: PyReadonlyArray1<'_, f64>,
        b: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        Ok(self.inner.distance(a.as_slice()?, b.as_slice()?))
    }
}
