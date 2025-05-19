#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};

use augurs_seasons::{Detector, PeriodogramDetector};
use augurs_testing::data::SEASON_EIGHT;

fn season_eight(c: &mut Criterion) {
    let y = SEASON_EIGHT;
    let detector = PeriodogramDetector::builder().build();
    c.bench_function("season_eight", |b| {
        b.iter(|| detector.detect(y));
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = season_eight
}
criterion_main!(benches);
