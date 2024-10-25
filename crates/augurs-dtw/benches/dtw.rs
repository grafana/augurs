#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;

use augurs_dtw::Dtw;

fn examples() -> Vec<Vec<f64>> {
    let raw = include_str!("../data/series.csv");
    let n_columns = raw.lines().next().unwrap().split(',').count();
    let n_rows = raw.lines().count();
    let mut examples = vec![Vec::with_capacity(n_rows); n_columns];
    for line in raw.lines() {
        for (i, value) in line.split(',').enumerate() {
            let value: f64 = value.parse().unwrap();
            if !value.is_nan() {
                examples[i].push(value);
            }
        }
    }
    examples
}

fn distance_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_euclidean");
    let examples = examples();
    let (s, t) = (&examples[0], &examples[1]);
    let windows = [None, Some(2), Some(5), Some(10), Some(20), Some(50)];
    for window in windows {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", window)),
            &(s, t),
            |b, (s, t)| {
                b.iter(|| {
                    let mut dtw = Dtw::euclidean();
                    if let Some(window) = window {
                        dtw = dtw.with_window(window);
                    }
                    dtw.distance(s, t)
                });
            },
        );
    }
}

fn distance_matrix_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrix_euclidean");
    let examples = examples();
    let examples = examples.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let windows = [Some(2), Some(10)];
    let parallelize = [false, true];
    for (window, parallelize) in windows.into_iter().cartesian_product(parallelize) {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "window: {:?}, parallelize: {:?}",
                window, parallelize
            )),
            &examples,
            |b, examples| {
                b.iter(|| {
                    let mut dtw = Dtw::euclidean().parallelize(parallelize);
                    if let Some(window) = window {
                        dtw = dtw.with_window(window).with_max_distance(window as f64);
                    }
                    dtw.distance_matrix(examples)
                });
            },
        );
    }
}

criterion_group!(benches, distance_euclidean, distance_matrix_euclidean);
criterion_main!(benches);
