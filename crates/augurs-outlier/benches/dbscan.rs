#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

use augurs_outlier::{DbscanDetector, OutlierDetector};

/// The shapes to benchmark, as `(n_series, n_timestamps)`.
///
/// DBSCAN outlier detection is run per timestamp over the values of every series,
/// so both dimensions matter independently: the number of series sets the size of
/// each inner sort and scan, and the number of timestamps sets how many times that
/// work happens. These shapes cover "a few series over a long window" through to
/// "very many series over a short window".
const SHAPES: &[(usize, usize)] = &[(10, 1000), (50, 1000), (200, 500), (1000, 200)];

/// Generate `n_series` aligned series of `n_timestamps` values each.
///
/// Most series follow a shared baseline, while a tenth of them drift away from it
/// for stretches at a time, so series repeatedly start and stop being outliers.
/// That keeps the outlier interval bookkeeping in `detect` honest: data where the
/// set of outliers never changes wouldn't exercise it.
fn generate(n_series: usize, n_timestamps: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(42);
    let baseline: Vec<f64> = (0..n_timestamps)
        .map(|i| 100.0 + 10.0 * (i as f64 / 50.0).sin())
        .collect();
    (0..n_series)
        .map(|series| {
            let outlying = series % 10 == 0;
            (0..n_timestamps)
                .map(|i| {
                    let noise: f64 = rng.gen_range(-0.2..0.2);
                    let drift = if outlying && (i / 20) % 3 == 0 {
                        50.0
                    } else {
                        0.0
                    };
                    baseline[i] + noise + drift
                })
                .collect()
        })
        .collect()
}

fn shape_id(n_series: usize, n_timestamps: usize) -> BenchmarkId {
    BenchmarkId::from_parameter(format!("{n_series}x{n_timestamps}"))
}

/// Transposing and sorting the input, which `OutlierDetector::preprocess` does once
/// and `detect` can then be re-run against with different sensitivities.
fn dbscan_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan_preprocess");
    for &(n_series, n_timestamps) in SHAPES {
        let series = generate(n_series, n_timestamps);
        let refs: Vec<&[f64]> = series.iter().map(|s| s.as_slice()).collect();
        let detector = DbscanDetector::with_sensitivity(0.5).unwrap();
        group.bench_with_input(shape_id(n_series, n_timestamps), &refs, |b, refs| {
            b.iter(|| detector.preprocess(refs).unwrap());
        });
    }
    group.finish();
}

/// The detection itself: the per-timestamp 1d DBSCAN plus the outlier interval
/// bookkeeping across timestamps.
fn dbscan_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan_detect");
    for &(n_series, n_timestamps) in SHAPES {
        let series = generate(n_series, n_timestamps);
        let refs: Vec<&[f64]> = series.iter().map(|s| s.as_slice()).collect();
        let detector = DbscanDetector::with_sensitivity(0.5).unwrap();
        let preprocessed = detector.preprocess(&refs).unwrap();
        group.bench_with_input(
            shape_id(n_series, n_timestamps),
            &preprocessed,
            |b, preprocessed| {
                b.iter(|| detector.detect(preprocessed).unwrap());
            },
        );
    }
    group.finish();
}

/// Sweep sensitivity at a fixed shape.
///
/// A high sensitivity gives a small epsilon, which means no cluster is found and
/// every series is an outlier; a low sensitivity gives a cluster that spans nearly
/// all the values. These are the two extremes of the per-timestamp scan and of the
/// interval bookkeeping, so both want covering.
fn dbscan_sensitivity(c: &mut Criterion) {
    let (n_series, n_timestamps) = (50, 1000);
    let series = generate(n_series, n_timestamps);
    let refs: Vec<&[f64]> = series.iter().map(|s| s.as_slice()).collect();

    let mut group = c.benchmark_group("dbscan_sensitivity");
    for sensitivity in [0.1, 0.5, 0.9, 0.99] {
        let detector = DbscanDetector::with_sensitivity(sensitivity).unwrap();
        let preprocessed = detector.preprocess(&refs).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(sensitivity),
            &preprocessed,
            |b, preprocessed| {
                b.iter(|| detector.detect(preprocessed).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    dbscan_preprocess,
    dbscan_detect,
    dbscan_sensitivity
);
criterion_main!(benches);
