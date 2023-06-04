use iai::{black_box, main};

use augurs_ets::{
    model::{ErrorComponent, ModelType, SeasonalComponent::None, TrendComponent, Unfit},
    AutoETS,
};
use augurs_testing::data::AIR_PASSENGERS as AP;

fn auto_fit() {
    AutoETS::new(1, "ZZN").unwrap().fit(black_box(AP)).unwrap();
}

fn fit() {
    Unfit::new(ModelType {
        error: ErrorComponent::Additive,
        trend: TrendComponent::Additive,
        season: None,
    })
    .damped(true)
    .fit(black_box(AP))
    .unwrap();
}

// We can't benchmark predict yet because iai doesn't have
// support for setup of benchmark functions.
// fn predict() {
//     let model = Unfit::new(ModelType {
//         error: ErrorComponent::Additive,
//         trend: TrendComponent::Additive,
//         season: None,
//     })
//     .damped(true)
//     .fit(AP)
//     .unwrap();
//     model.predict(24, 0.95).ok();
// }

main!(fit, auto_fit);
