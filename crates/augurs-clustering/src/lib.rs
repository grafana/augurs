#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use std::collections::VecDeque;

pub use augurs_core::DistanceMatrix;

/// DBSCAN clustering algorithm.
#[derive(Debug)]
pub struct Dbscan {
    epsilon: f64,
    min_cluster_size: usize,
}

impl Dbscan {
    /// Create a new DBSCAN instance clustering instance.
    ///
    /// # Arguments
    /// * `epsilon` - The maximum distance between two samples for one to be considered as in the
    ///     neighborhood of the other.
    /// * `min_cluster_size` - The number of samples in a neighborhood for a point to be considered as a core
    ///     point.
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
    /// The return value is a vector of cluster assignments, with `-1` indicating noise.
    pub fn fit(&self, distance_matrix: &DistanceMatrix) -> Vec<isize> {
        let n = distance_matrix.shape().0;
        let mut clusters = vec![-1; n];
        let mut cluster = 0;
        let mut visited = vec![false; n];
        let mut to_visit = VecDeque::with_capacity(n);

        // We'll reuse this vector to avoid reallocations.
        let mut neighbours = Vec::with_capacity(n);

        for (i, d) in distance_matrix.iter().enumerate() {
            // Skip if already assigned to a cluster.
            if clusters[i] != -1 {
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
                if clusters[neighbour] == -1 {
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
            cluster += 1;
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

        let clusters = Dbscan::new(0.5, 2).fit(&distance_matrix);
        assert_eq!(clusters, vec![-1, -1, -1, -1]);

        let clusters = Dbscan::new(1.0, 2).fit(&distance_matrix);
        assert_eq!(clusters, vec![0, 0, -1, -1]);

        let clusters = Dbscan::new(1.0, 3).fit(&distance_matrix);
        assert_eq!(clusters, vec![-1, -1, -1, -1]);

        let clusters = Dbscan::new(2.0, 2).fit(&distance_matrix);
        assert_eq!(clusters, vec![0, 0, 0, -1]);

        let clusters = Dbscan::new(2.0, 3).fit(&distance_matrix);
        assert_eq!(clusters, vec![0, 0, 0, -1]);

        let clusters = Dbscan::new(3.0, 3).fit(&distance_matrix);
        assert_eq!(clusters, vec![0, 0, 0, 0]);
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
        let clusters = Dbscan::new(10.0, 3).fit(&distance_matrix);
        let expected = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 2, -1, 2, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(clusters, expected);
    }
}
