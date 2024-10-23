//! Bindings for clustering algorithms.

use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_core_js::DistanceMatrix;

/// Options for the dynamic time warping calculation.
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct DbscanOptions {
    /// The maximum distance between two samples for one to be considered as in the
    /// neighborhood of the other.
    pub epsilon: f64,

    /// The number of samples in a neighborhood for a point to be considered as a core
    /// point.
    pub min_cluster_size: usize,
}

/// A DBSCAN clustering algorithm.
#[derive(Debug)]
#[wasm_bindgen]
pub struct DbscanClusterer {
    inner: augurs_clustering::DbscanClusterer,
}

#[wasm_bindgen]
impl DbscanClusterer {
    /// Create a new DBSCAN instance.
    #[wasm_bindgen(constructor)]
    pub fn new(opts: DbscanOptions) -> Self {
        Self {
            inner: augurs_clustering::DbscanClusterer::new(opts.epsilon, opts.min_cluster_size),
        }
    }

    /// Fit the DBSCAN clustering algorithm to the given distance matrix.
    ///
    /// The distance matrix can be obtained using the `Dtw` class.
    ///
    /// The return value is an `Int32Array` of cluster IDs, with `-1` indicating noise.
    #[wasm_bindgen]
    #[allow(non_snake_case)]
    pub fn fit(&self, distanceMatrix: &DistanceMatrix) -> Vec<isize> {
        self.inner.fit(distanceMatrix.inner())
    }
}
