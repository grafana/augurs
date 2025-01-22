#[allow(warnings)]
#[rustfmt::skip]
mod bindings;

use augurs_clustering::DbscanClusterer;
use bindings::{
    augurs::core::types::{DistanceMatrix, Error},
    exports::augurs::clustering::dbscan::{
        DbscanCluster, Guest, GuestDbscan, Options as DbscanOptions,
    },
};

impl From<augurs_clustering::DbscanCluster> for DbscanCluster {
    fn from(value: augurs_clustering::DbscanCluster) -> Self {
        match value {
            augurs_clustering::DbscanCluster::Noise => Self::Noise,
            augurs_clustering::DbscanCluster::Cluster(id) => Self::Cluster(id.into()),
        }
    }
}

struct Component;

impl Guest for Component {
    type Dbscan = Dbscan;
}

struct Dbscan {
    inner: DbscanClusterer,
}

impl GuestDbscan for Dbscan {
    fn new(options: DbscanOptions) -> Self {
        Self {
            inner: DbscanClusterer::new(options.epsilon, options.minimum_cluster_size as usize),
        }
    }

    fn fit(&self, distance_matrix: DistanceMatrix) -> Result<Vec<DbscanCluster>, Error> {
        let matrix = augurs_core::DistanceMatrix::try_from_square(distance_matrix)
            .map_err(|_| Error::InvalidDistanceMatrix)?;
        Ok(self
            .inner
            .fit(&matrix)
            .into_iter()
            .map(Into::into)
            .collect())
    }
}

bindings::export!(Component with_types_in bindings);
