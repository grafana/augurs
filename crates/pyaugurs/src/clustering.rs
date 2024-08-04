//! Bindings for clustering algorithms.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

/// A distance matrix.
///
/// This can be passed in several different formats:
/// - As a 2D list of lists of f64.
/// - As a 2D numpy array of f64.
/// - As an `augurs_core::DistanceMatrix`.
#[derive(Debug, FromPyObject)]
pub enum InputDistanceMatrix<'py> {
    /// A list of lists of f64.
    Lists(Vec<Vec<f64>>),
    /// A 2D numpy array of f64.
    Numpy(PyReadonlyArray2<'py, f64>),
    /// An `augurs_core::DistanceMatrix`.
    Augurs(crate::distance::DistanceMatrix),
}

impl TryFrom<InputDistanceMatrix<'_>> for augurs_core::DistanceMatrix {
    type Error = PyErr;
    fn try_from(val: InputDistanceMatrix<'_>) -> Result<Self, Self::Error> {
        match val {
            InputDistanceMatrix::Augurs(matrix) => Ok(matrix.into()),
            InputDistanceMatrix::Lists(matrix) => {
                Ok(augurs_core::DistanceMatrix::try_from_square(matrix)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?)
            }
            InputDistanceMatrix::Numpy(matrix) => {
                let matrix: Vec<_> = matrix
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .chunks_exact(matrix.shape()[1])
                    .map(|x| x.to_vec())
                    .collect();
                Ok(augurs_core::DistanceMatrix::try_from_square(matrix)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?)
            }
        }
    }
}

/// Dbscan clustering.
///
/// :param eps: the maximum distance between two samples for one to be considered as in the
///     neighborhood of the other.
/// :param min_cluster_size: the number of samples in a neighborhood for a point to be considered as a core
///     point.
#[derive(Debug)]
#[pyclass]
pub struct Dbscan {
    inner: augurs_clustering::Dbscan,
}

#[pymethods]
impl Dbscan {
    fn __repr__(&self) -> String {
        format!(
            "Dbscan(eps={}, min_cluster_size={})",
            self.inner.epsilon(),
            self.inner.min_cluster_size(),
        )
    }

    #[new]
    fn new(eps: f64, min_cluster_size: usize) -> Dbscan {
        Dbscan {
            inner: augurs_clustering::Dbscan::new(eps, min_cluster_size),
        }
    }

    /// Fit the DBSCAN clustering algorithm to the given distance matrix, represented as numpy
    /// array.
    ///
    /// :param distance_matrix: distance matrix between samples. Can be either:
    ///    - a 2D square numpy array
    ///    - a list of lists of floats
    ///    - an `augurs::DistanceMatrix`
    /// :return: the cluster assignments, with `-1` indicating noise.
    pub fn fit(
        &self,
        py: Python<'_>,
        distance_matrix: InputDistanceMatrix<'_>,
    ) -> PyResult<Py<PyArray1<isize>>> {
        let distance_matrix = distance_matrix.try_into()?;
        Ok(self
            .inner
            .fit(&distance_matrix)
            .into_pyarray_bound(py)
            .into())
    }
}
