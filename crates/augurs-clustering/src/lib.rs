#![doc = include_str!("../README.md")]

use std::{collections::VecDeque, num::NonZeroU32};

pub use augurs_core::DistanceMatrix;

/// A cluster identified by the DBSCAN algorithm.
///
/// This is either a noise cluster, or a cluster with a specific ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DbscanCluster {
    /// A noise cluster.
    Noise,
    /// A cluster with the given ID.
    ///
    /// The ID is not guaranteed to remain the same between runs of the algorithm.
    ///
    /// We use a `NonZeroU32` here to ensure that the ID is never zero. This is mostly
    /// just a size optimization.
    Cluster(NonZeroU32),
}

impl DbscanCluster {
    /// Returns true if this cluster is a noise cluster.
    pub fn is_noise(&self) -> bool {
        matches!(self, Self::Noise)
    }

    /// Returns true if this cluster is a cluster with the given ID.
    pub fn is_cluster(&self) -> bool {
        matches!(self, Self::Cluster(_))
    }

    /// Returns the ID of the cluster, if it is a cluster, or `-1` if it is a noise cluster.
    pub fn as_i32(&self) -> i32 {
        match self {
            Self::Noise => -1,
            Self::Cluster(id) => id.get() as i32,
        }
    }

    fn increment(&mut self) {
        match self {
            Self::Noise => unreachable!(),
            Self::Cluster(id) => *id = id.checked_add(1).expect("cluster ID overflow"),
        }
    }
}

// Simplify tests by allowing comparisons with i32.
#[cfg(test)]
impl PartialEq<i32> for DbscanCluster {
    fn eq(&self, other: &i32) -> bool {
        if self.is_noise() {
            *other == -1
        } else {
            self.as_i32() == *other
        }
    }
}

/// DBSCAN clustering algorithm.
#[derive(Debug)]
pub struct DbscanClusterer {
    epsilon: f64,
    min_cluster_size: usize,
}

impl DbscanClusterer {
    /// Create a new DBSCAN instance clustering instance.
    ///
    /// # Arguments
    /// * `epsilon` - The maximum distance between two samples for one to be considered as in the
    ///   neighborhood of the other.
    /// * `min_cluster_size` - The number of samples in a neighborhood for a point to be considered as a core
    ///   point.
    pub fn new(epsilon: f64, min_cluster_size: usize) -> Self {
        Self {
            epsilon,
            min_cluster_size,
        }
    }

    /// Return epsilon, the maximum distance between two samples for one to be considered as in the
    /// neighborhood of the other.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Return the minimum number of points in a neighborhood for a point to be considered as a core
    /// point.
    pub fn min_cluster_size(&self) -> usize {
        self.min_cluster_size
    }

    /// Run the DBSCAN clustering algorithm.
    ///
    /// The return value is a vector of cluster assignments, with `DbscanCluster::Noise` indicating noise.
    pub fn fit(&self, distance_matrix: &DistanceMatrix) -> Vec<DbscanCluster> {
        let n = distance_matrix.shape().0;
        let mut clusters = vec![DbscanCluster::Noise; n];
        let mut cluster = DbscanCluster::Cluster(NonZeroU32::new(1).unwrap());
        let mut visited = vec![false; n];
        let mut to_visit = VecDeque::with_capacity(n);

        // We'll reuse this vector to avoid reallocations.
        let mut neighbours = Vec::with_capacity(n);

        for (i, d) in distance_matrix.iter().enumerate() {
            // Skip if already assigned to a cluster.
            if clusters[i].is_cluster() {
                continue;
            }
            self.find_neighbours(i, d, &mut neighbours);
            if neighbours.len() < self.min_cluster_size - 1 {
                // Not a core point: leave marked as noise.
                continue;
            }
            // We're in a cluster: expand it to all core neighbours.
            // Mark this point as visited so we can skip checking it later.
            visited[i] = true;
            clusters[i] = cluster;
            // Mark all noise neighbours as visited and add them to the queue.
            for neighbour in neighbours.drain(..) {
                if clusters[neighbour].is_noise() {
                    visited[neighbour] = true;
                    to_visit.push_back(neighbour);
                }
            }

            // Expand the cluster.
            while let Some(candidate) = to_visit.pop_front() {
                clusters[candidate] = cluster;
                self.find_neighbours(candidate, &distance_matrix[candidate], &mut neighbours);
                if neighbours.len() >= self.min_cluster_size - 1 {
                    // Add unvisited extended neighbours to the queue.
                    for neighbour in neighbours.drain(..) {
                        if !visited[neighbour] {
                            visited[neighbour] = true;
                            to_visit.push_back(neighbour);
                        }
                    }
                }
            }
            cluster.increment();
        }
        clusters
    }

    #[inline]
    fn find_neighbours(&self, i: usize, dists: &[f64], n: &mut Vec<usize>) {
        n.clear();
        n.extend(
            dists
                .iter()
                .enumerate()
                .filter(|(j, &x)| i != *j && x <= self.epsilon)
                .map(|(j, _)| j),
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dbscan() {
        let distance_matrix = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 0.0, 3.0, 3.0],
            vec![2.0, 3.0, 0.0, 4.0],
            vec![3.0, 3.0, 4.0, 0.0],
        ];
        let distance_matrix = DistanceMatrix::try_from_square(distance_matrix).unwrap();

        let clusters = DbscanClusterer::new(0.5, 2).fit(&distance_matrix);
        assert_eq!(clusters, vec![-1, -1, -1, -1]);

        let clusters = DbscanClusterer::new(1.0, 2).fit(&distance_matrix);
        assert_eq!(clusters, vec![1, 1, -1, -1]);

        let clusters = DbscanClusterer::new(1.0, 3).fit(&distance_matrix);
        assert_eq!(clusters, vec![-1, -1, -1, -1]);

        let clusters = DbscanClusterer::new(2.0, 2).fit(&distance_matrix);
        assert_eq!(clusters, vec![1, 1, 1, -1]);

        let clusters = DbscanClusterer::new(2.0, 3).fit(&distance_matrix);
        assert_eq!(clusters, vec![1, 1, 1, -1]);

        let clusters = DbscanClusterer::new(3.0, 3).fit(&distance_matrix);
        assert_eq!(clusters, vec![1, 1, 1, 1]);
    }

    #[test]
    fn dbscan_real() {
        let distance_matrix = include_str!("../data/dist.csv")
            .lines()
            .map(|l| {
                l.split(',')
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        let distance_matrix = DistanceMatrix::try_from_square(distance_matrix).unwrap();
        let clusters = DbscanClusterer::new(10.0, 3).fit(&distance_matrix);
        let expected = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 3, -1, 3, -1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        assert_eq!(clusters, expected);
    }
}
