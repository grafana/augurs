use iai::{black_box, main};

use augurs_mstl::{MSTLModel, NaiveTrend};
use augurs_testing::data::VIC_ELEC;

fn vic_elec_fit(y: Vec<f64>, periods: Vec<usize>, params: stlrs::StlParams) {
    MSTLModel::new(periods, NaiveTrend::new())
        .stl_params(params)
        .fit(y)
        .ok();
}

fn bench_vic_elec_fit() {
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
    vic_elec_fit(
        black_box(y.clone()),
        black_box(vec![24, 24 * 7]),
        black_box(stl_params),
    );
}

main!(bench_vic_elec_fit);
