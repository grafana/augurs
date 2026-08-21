#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use augurs_clustering::DbscanClusterer;
use augurs_core::DistanceMatrix;

fn distance_matrix() -> DistanceMatrix {
    let distance_matrix = include_str!("../data/dist.csv")
        .lines()
        .map(|l| {
            l.split(',')
                .map(|s| s.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    DistanceMatrix::try_from_square(distance_matrix).unwrap()
}

fn dbscan(c: &mut Criterion) {
    let distance_matrix = distance_matrix();
    c.bench_function("dbscan", |b| {
        b.iter(|| {
            DbscanClusterer::new(10.0, 3).fit(&distance_matrix);
        });
    });
}

/// `fit` spends nearly all of its time in `find_neighbours`, whose cost depends on
/// how many points fall within `epsilon`. Sweep epsilon so that regressions in that
/// loop are visible across the whole range of neighbour counts, rather than only at
/// the one value the `dbscan` benchmark happens to use.
fn dbscan_epsilon(c: &mut Criterion) {
    let distance_matrix = distance_matrix();
    let mut group = c.benchmark_group("dbscan_epsilon");
    for epsilon in [1.0, 5.0, 10.0, 20.0, 50.0] {
        group.bench_with_input(
            BenchmarkId::from_parameter(epsilon),
            &epsilon,
            |b, &epsilon| {
                b.iter(|| DbscanClusterer::new(epsilon, 3).fit(&distance_matrix));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, dbscan, dbscan_epsilon);
criterion_main!(benches);
