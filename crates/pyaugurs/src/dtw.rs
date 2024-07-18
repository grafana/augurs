//! Bindings for Dynamic Time Warping (DTW).

use std::{fmt, str::FromStr};

use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};

use augurs_core::DistanceMatrix;
use augurs_dtw::{Euclidean, Manhattan};

enum InnerDtw {
    Euclidean(augurs_dtw::Dtw<Euclidean>),
    Manhattan(augurs_dtw::Dtw<Manhattan>),
}

impl InnerDtw {
    fn window(&self) -> Option<usize> {
        match self {
            InnerDtw::Euclidean(dtw) => dtw.window(),
            InnerDtw::Manhattan(dtw) => dtw.window(),
        }
    }

    fn with_window(self, window: usize) -> Self {
        match self {
            InnerDtw::Euclidean(dtw) => InnerDtw::Euclidean(dtw.with_window(window)),
            InnerDtw::Manhattan(dtw) => InnerDtw::Manhattan(dtw.with_window(window)),
        }
    }

    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            InnerDtw::Euclidean(dtw) => dtw.distance(a, b),
            InnerDtw::Manhattan(dtw) => dtw.distance(a, b),
        }
    }

    fn distance_matrix(&self, series: &[&[f64]]) -> DistanceMatrix {
        match self {
            InnerDtw::Euclidean(dtw) => dtw.distance_matrix(series),
            InnerDtw::Manhattan(dtw) => dtw.distance_matrix(series),
        }
    }
}

impl fmt::Debug for InnerDtw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InnerDtw::Euclidean(_) => write!(f, "InnerDtw::Euclidean"),
            InnerDtw::Manhattan(_) => write!(f, "InnerDtw::Manhattan"),
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
enum DistanceFn {
    #[default]
    Euclidean,
    Manhattan,
}

impl FromStr for DistanceFn {
    type Err = PyErr;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "euclidean" => Ok(DistanceFn::Euclidean),
            "manhattan" => Ok(DistanceFn::Manhattan),
            _ => Err(PyValueError::new_err(format!(
                "Invalid distance function: {}",
                s
            ))),
        }
    }
}

/// Dynamic Time Warping implementation.
///
/// The `window` parameter can be used to specify the Sakoe-Chiba band size.
///
/// This will use the Euclidean distance by default. You can also use the Manhattan distance by
/// passing `distance_fn="manhattan"`.
///
/// ```python
/// import numpy as np
/// from augurs import Dtw
///
/// a = np.array([0.0, 1.0, 2.0])
/// b = np.array([3.0, 4.0, 5.0])
/// distance = Dtw(window=2).distance(a, b)
/// distance = Dtw(window=2, distance_fn="manhattan").distance(a, b)
/// ```
#[derive(Debug)]
#[pyclass]
pub struct Dtw {
    inner: InnerDtw,
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
    fn new(window: Option<usize>, distance_fn: Option<&str>) -> PyResult<Self> {
        let mut inner = match distance_fn
            .map(|x| x.parse())
            .transpose()?
            .unwrap_or_default()
        {
            DistanceFn::Euclidean => InnerDtw::Euclidean(augurs_dtw::Dtw::euclidean()),
            DistanceFn::Manhattan => InnerDtw::Manhattan(augurs_dtw::Dtw::manhattan()),
        };
        if let Some(window) = window {
            inner = inner.with_window(window);
        }
        Ok(Self { inner })
    }

    /// Calculate the distance between two arrays under Dynamic Time Warping.
    pub fn distance(
        &self,
        a: PyReadonlyArray1<'_, f64>,
        b: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        Ok(self.inner.distance(a.as_slice()?, b.as_slice()?))
    }

    /// Calculate the pairwise distance matrix between a list of arrays under Dynamic Time Warping.
    pub fn distance_matrix(
        &self,
        series: Vec<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let series: Vec<_> = series
            .iter()
            .map(|a| a.as_slice())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.inner.distance_matrix(&series).into_inner())
    }
}
