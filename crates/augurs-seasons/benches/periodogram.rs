use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

use augurs_seasons::{Detector, PeriodogramDetector};
use augurs_testing::data::SEASON_EIGHT;

fn season_eight(c: &mut Criterion) {
    let y = SEASON_EIGHT;
    c.bench_function("season_eight", |b| {
        b.iter(|| {
            PeriodogramDetector::builder()
                .build(y)
                .detect()
                .collect::<Vec<_>>()
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Protobuf));
    targets = season_eight
}
criterion_main!(benches);
