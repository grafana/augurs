//! Distance matrix module for PyAugurs.
//!
//! We define our own wrapper around the `augurs_core` distance matrix type
//! which can be returned to Python and converted to a numpy array, or passed
//! back into Rust without the conversion cost.

use numpy::{ndarray::Array2, IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// A distance matrix.
///
/// This is intentionally opaque; it can either be passed back to other `augurs`
/// functions or converted to a numpy array using `to_numpy`.
#[derive(Debug, Clone)]
#[pyclass]
pub struct DistanceMatrix {
    inner: augurs_core::DistanceMatrix,
}

impl From<augurs_core::DistanceMatrix> for DistanceMatrix {
    fn from(inner: augurs_core::DistanceMatrix) -> Self {
        Self { inner }
    }
}

impl From<DistanceMatrix> for augurs_core::DistanceMatrix {
    fn from(matrix: DistanceMatrix) -> Self {
        matrix.inner
    }
}

#[pymethods]
impl DistanceMatrix {
    fn __repr__(&self) -> String {
        format!("DistanceMatrix(shape={:?})", self.inner.shape())
    }

    /// Convert the distance matrix to a numpy array.
    ///
    /// The resulting array will be a 2D array of f64.
    // This currently requires some copying, as the distance matrix is stored
    // in a nested `Vec<Vec<f64>>` format rather than as a contiguous array.
    // We could potentially optimize this by storing the distance matrix in
    // an `Array2<f64>`, but this would require some changes to the core
    // library.
    pub fn to_numpy(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        let mut arr = Array2::zeros(self.inner.shape());
        for (vec_row, mut arr_row) in self.inner.iter().zip(arr.outer_iter_mut()) {
            for (val, elem) in vec_row.iter().zip(arr_row.iter_mut()) {
                *elem = *val;
            }
        }
        arr.into_pyarray(py).into()
    }
}

impl DistanceMatrix {
    pub fn inner(&self) -> &augurs_core::DistanceMatrix {
        &self.inner
    }
}
