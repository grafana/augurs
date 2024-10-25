#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};

use augurs_clustering::DbscanClusterer;
use augurs_core::DistanceMatrix;

fn dbscan(c: &mut Criterion) {
    let distance_matrix = include_str!("../data/dist.csv")
        .lines()
        .map(|l| {
            l.split(',')
                .map(|s| s.parse::<f64>().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    let distance_matrix = DistanceMatrix::try_from_square(distance_matrix).unwrap();
    c.bench_function("dbscan", |b| {
        b.iter(|| {
            DbscanClusterer::new(10.0, 3).fit(&distance_matrix);
        });
    });
}

criterion_group!(benches, dbscan);
criterion_main!(benches);
