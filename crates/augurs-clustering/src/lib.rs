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

    /// Collect the indices of all points within `epsilon` of point `i`.
    ///
    /// This is the hot loop of [`Self::fit`]: it is called roughly once per point,
    /// so a whole clustering run is `O(n^2)` in this function.
    ///
    /// It's written in a "branchless" style: rather than conditionally pushing
    /// matching indices, every index is written to the buffer unconditionally and
    /// the cursor is only advanced when the predicate holds. This keeps the loop
    /// body free of both a data-dependent branch and the per-push capacity check
    /// that `Vec::extend` performs, and measures ~2x faster than the equivalent
    /// `filter`/`extend` across the full range of neighbour counts.
    #[inline]
    fn find_neighbours(&self, i: usize, dists: &[f64], n: &mut Vec<usize>) {
        // Reserve room for every index without initializing it. `resize` would
        // zero-fill the whole buffer even though every slot the loop below
        // touches gets overwritten anyway, which just wastes a full pass over
        // it on every call.
        n.clear();
        n.reserve(dists.len());
        let spare = n.spare_capacity_mut();
        let mut found = 0;
        for (j, &dist) in dists.iter().enumerate() {
            // SAFETY: after `j` iterations `found` has advanced by at most one
            // per iteration, so `found <= j < dists.len() <= spare.len()`.
            // `get_unchecked_mut` (rather than indexing `spare`) keeps this
            // write free of the bounds-check branch that indexing would add,
            // which is the whole point of writing it "branchlessly" below.
            unsafe { spare.get_unchecked_mut(found).write(j) };
            // Note the non-short-circuiting `&`: both operands are cheap, and
            // avoiding the branch is the entire point.
            found += ((dist <= self.epsilon) & (i != j)) as usize;
        }
        // SAFETY: indices `0..found` were each written exactly once above, in
        // order, since `found` only ever advances by one slot at a time.
        unsafe { n.set_len(found) };
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
    fn find_neighbours() {
        let dbscan = DbscanClusterer::new(2.0, 2);
        let mut neighbours = Vec::new();

        // The point itself is never its own neighbour, and indices come back in
        // ascending order.
        dbscan.find_neighbours(1, &[1.0, 0.0, 5.0, 2.0, 3.0], &mut neighbours);
        assert_eq!(neighbours, vec![0, 3]);

        // No neighbours at all: the buffer must end up empty, not merely truncated
        // to something stale from a previous call.
        dbscan.find_neighbours(0, &[0.0, 9.0, 9.0], &mut neighbours);
        assert!(neighbours.is_empty());

        // Every other point is a neighbour.
        dbscan.find_neighbours(2, &[1.0, 1.0, 0.0, 1.0], &mut neighbours);
        assert_eq!(neighbours, vec![0, 1, 3]);

        // Exactly epsilon away counts as a neighbour.
        dbscan.find_neighbours(0, &[0.0, 2.0, 2.000001], &mut neighbours);
        assert_eq!(neighbours, vec![1]);
    }

    // `find_neighbours` writes into spare capacity via `get_unchecked_mut` and
    // `set_len` rather than safe indexing, so its output is checked here
    // against a naive filter across many shapes and epsilons, including the
    // buffer being reused (and so already containing stale data) between
    // calls, to guard against the unsafe rewrite ever reading or writing out
    // of bounds or leaving stale entries behind.
    #[test]
    fn find_neighbours_matches_naive_filter() {
        // Small xorshift PRNG so the sweep is deterministic without adding a
        // dependency just for this test.
        struct Xorshift(u64);
        impl Xorshift {
            fn next_u64(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            // A small, low-cardinality range of distances so that plenty of
            // trials land exactly on the epsilon boundary.
            fn next_dist(&mut self) -> f64 {
                (self.next_u64() % 10) as f64
            }
        }
        let mut rng = Xorshift(0x9E3779B97F4A7C15);
        let mut neighbours = Vec::new();

        for trial in 0..2_000 {
            let len = (rng.next_u64() % 20) as usize;
            let dists: Vec<f64> = (0..len).map(|_| rng.next_dist()).collect();
            let i = if len == 0 { 0 } else { (rng.next_u64() as usize) % len };
            let epsilon = rng.next_dist();
            let dbscan = DbscanClusterer::new(epsilon, 2);

            dbscan.find_neighbours(i, &dists, &mut neighbours);

            let expected: Vec<usize> = dists
                .iter()
                .enumerate()
                .filter(|&(j, &d)| d <= epsilon && j != i)
                .map(|(j, _)| j)
                .collect();

            assert_eq!(
                neighbours, expected,
                "trial {trial}: dists={dists:?}, i={i}, epsilon={epsilon}"
            );
        }
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
