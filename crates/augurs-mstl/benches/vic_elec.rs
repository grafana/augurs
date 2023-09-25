use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use pprof::criterion::{Output, PProfProfiler};

use augurs_mstl::{MSTLModel, NaiveTrend};
use augurs_testing::data::VIC_ELEC;

fn vic_elec(c: &mut Criterion) {
    let y = &*VIC_ELEC;
    let mut stl_params = stlrs::params();
    stl_params
        .seasonal_degree(0)
        .seasonal_jump(1)
        .trend_degree(1)
        .trend_jump(1)
        .low_pass_degree(1)
        .low_pass_degree(1)
        .inner_loops(2)
        .outer_loops(0);
    let mut mstl_params = stlrs::MstlParams::new();
    mstl_params.stl_params(stl_params);
    c.bench_function("vic_elec", |b| {
        b.iter_batched(
            || (y.clone(), vec![24, 24 * 7], mstl_params.clone()),
            |(y, periods, stl_params)| {
                MSTLModel::new(periods, NaiveTrend::new())
                    .mstl_params(stl_params)
                    .fit(&y)
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Protobuf));
    targets = vic_elec
}
criterion_main!(benches);
