//! Bindings for Dynamic Time Warping (DTW).

use std::fmt;

use js_sys::Float64Array;
use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_dtw::{Euclidean, Manhattan};

enum InnerDtw {
    Euclidean(augurs_dtw::Dtw<Euclidean>),
    Manhattan(augurs_dtw::Dtw<Manhattan>),
}

impl InnerDtw {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            InnerDtw::Euclidean(dtw) => dtw.distance(a, b),
            InnerDtw::Manhattan(dtw) => dtw.distance(a, b),
        }
    }

    fn distance_matrix(&self, series: &[&[f64]]) -> DistanceMatrix {
        match self {
            InnerDtw::Euclidean(dtw) => dtw.distance_matrix(series).into(),
            InnerDtw::Manhattan(dtw) => dtw.distance_matrix(series).into(),
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

/// Options for the dynamic time warping calculation.
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct DtwOptions {
    /// The size of the Sakoe-Chiba band.
    #[tsify(optional)]
    pub window: Option<usize>,

    /// The maximum distance permitted between two points.
    ///
    /// If the distance between two points exceeds this value, the algorithm will
    /// early abandon and use `maxDistance`.
    ///
    /// Only used when calculating distance matrices using [`Dtw::distanceMatrix`],
    /// not when calculating the distance between two series.
    #[tsify(optional)]
    pub max_distance: Option<f64>,

    /// The lower bound, used for early abandoning.
    /// If specified, before calculating the DTW (which can be expensive), check if the
    /// lower bound of the DTW is greater than this distance; if so, skip the DTW
    /// calculation and return this bound instead.
    #[tsify(optional)]
    pub lower_bound: Option<f64>,

    /// The upper bound, used for early abandoning.
    /// If specified, before calculating the DTW (which can be expensive), check if the
    /// upper bound of the DTW is less than this distance; if so, skip the DTW
    /// calculation and return this bound instead.
    #[tsify(optional)]
    pub upper_bound: Option<f64>,

    #[cfg(feature = "parallel")]
    /// Parallelize the DTW distance matrix calculation.
    #[tsify(optional)]
    pub parallelize: Option<bool>,
}

/// A distance matrix.
///
/// This is intentionally opaque; it should only be passed back to `augurs` for further processing,
/// e.g. to calculate nearest neighbors or perform clustering.
#[derive(Debug)]
#[wasm_bindgen]
pub struct DistanceMatrix {
    inner: augurs_core::DistanceMatrix,
}

impl DistanceMatrix {
    pub fn inner(&self) -> &augurs_core::DistanceMatrix {
        &self.inner
    }
}

impl From<augurs_core::DistanceMatrix> for DistanceMatrix {
    fn from(inner: augurs_core::DistanceMatrix) -> Self {
        Self { inner }
    }
}

impl From<DistanceMatrix> for augurs_core::DistanceMatrix {
    fn from(matrix: DistanceMatrix) -> Self {
        Self::try_from_square(matrix.inner.into_inner()).unwrap()
    }
}

/// Dynamic Time Warping.
///
/// The `window` parameter can be used to specify the Sakoe-Chiba band size.
/// The distance function depends on the constructor used; `euclidean` and
/// `manhattan` are available, `euclidean` being the default.
#[derive(Debug)]
#[wasm_bindgen]
pub struct Dtw {
    inner: InnerDtw,
}

#[wasm_bindgen]
impl Dtw {
    /// Create a new `Dtw` instance using the Euclidean distance.
    #[wasm_bindgen]
    pub fn euclidean(opts: Option<DtwOptions>) -> Result<Dtw, JsValue> {
        let opts = opts.unwrap_or_default();
        let mut dtw = augurs_dtw::Dtw::euclidean();
        if let Some(window) = opts.window {
            dtw = dtw.with_window(window);
        }
        if let Some(max_distance) = opts.max_distance {
            dtw = dtw.with_max_distance(max_distance);
        }
        if let Some(lower_bound) = opts.lower_bound {
            dtw = dtw.with_lower_bound(lower_bound);
        }
        if let Some(upper_bound) = opts.upper_bound {
            dtw = dtw.with_upper_bound(upper_bound);
        }
        #[cfg(feature = "parallel")]
        if let Some(parallelize) = opts.parallelize {
            dtw = dtw.parallelize(parallelize);
        }
        Ok(Dtw {
            inner: InnerDtw::Euclidean(dtw),
        })
    }

    /// Create a new `Dtw` instance using the Euclidean distance.
    #[wasm_bindgen]
    pub fn manhattan(opts: Option<DtwOptions>) -> Result<Dtw, JsValue> {
        let opts = opts.unwrap_or_default();
        let mut dtw = augurs_dtw::Dtw::manhattan();
        if let Some(window) = opts.window {
            dtw = dtw.with_window(window);
        }
        if let Some(max_distance) = opts.max_distance {
            dtw = dtw.with_max_distance(max_distance);
        }
        if let Some(lower_bound) = opts.lower_bound {
            dtw = dtw.with_lower_bound(lower_bound);
        }
        if let Some(upper_bound) = opts.upper_bound {
            dtw = dtw.with_upper_bound(upper_bound);
        }
        Ok(Dtw {
            inner: InnerDtw::Manhattan(dtw),
        })
    }

    /// Calculate the distance between two arrays under Dynamic Time Warping.
    #[wasm_bindgen]
    pub fn distance(&self, a: Float64Array, b: Float64Array) -> f64 {
        self.inner.distance(&a.to_vec(), &b.to_vec())
    }

    /// Compute the distance matrix between all pairs of series.
    ///
    /// The series do not all have to be the same length.
    #[wasm_bindgen(js_name = distanceMatrix)]
    pub fn distance_matrix(&self, series: Vec<Float64Array>) -> DistanceMatrix {
        let vecs: Vec<_> = series.iter().map(|x| x.to_vec()).collect();
        let slices = vecs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        self.inner.distance_matrix(&slices)
    }
}
