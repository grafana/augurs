use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

use augurs_ets::{
    model::{ErrorComponent, ModelType, SeasonalComponent::None, TrendComponent, Unfit},
    AutoETS,
};
use augurs_testing::data::AIR_PASSENGERS as AP;

fn auto_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_fit");
    group.bench_function("air_passengers", |b| {
        b.iter_batched_ref(
            || AutoETS::new(1, "ZZN").unwrap(),
            |auto| {
                auto.fit(AP).unwrap();
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit");
    group.bench_function("air_passengers", |b| {
        b.iter_batched(
            || {
                Unfit::new(ModelType {
                    error: ErrorComponent::Additive,
                    trend: TrendComponent::Additive,
                    season: None,
                })
                .damped(true)
            },
            |model| model.fit(AP),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn forecast(c: &mut Criterion) {
    let model = Unfit::new(ModelType {
        error: ErrorComponent::Additive,
        trend: TrendComponent::Additive,
        season: None,
    })
    .damped(true)
    .fit(AP)
    .unwrap();
    let mut group = c.benchmark_group("forecast");
    group.bench_function("air_passengers", |b| {
        b.iter(|| {
            model.predict(24, 0.95);
        })
    });
}

criterion_group! {
    name = benches;
    // config = Criterion::default();
    config = Criterion::default().with_profiler(PProfProfiler::new(10000, Output::Protobuf));
    targets = auto_fit, fit, forecast,
}
criterion_main!(benches);
