//! This example demonstrates how to use the high level forecasting API
//! of augurs to forecast a time series using any model that implements
//! `Fit` and `Predict`.
//!
//! The forecaster can apply transformations to the input data before
//! fitting the model, and will then back-transform the predictions.

use augurs::{
    ets::AutoETS,
    forecaster::{
        transforms::{LinearInterpolator, Log, MinMaxScaler},
        Forecaster, Transformer,
    },
    mstl::MSTLModel,
};

const DATA: &[f64] = &[
    1.0,
    1.2,
    1.4,
    1.5,
    // Note the missing value represented as a `f64::NAN`.
    // This will be handled by the `LinearInterpolator` transform.
    f64::NAN,
    1.4,
    1.2,
    1.5,
    1.6,
    2.0,
    1.9,
    1.8,
];

fn main() {
    // Set up the model. We're going to use an MSTL model to handle
    // multiple seasonalities, with a non-seasonal `AutoETS` model
    // for the trend component.
    // We could also use any model that implements `augurs_core::Fit`.
    let ets = AutoETS::non_seasonal().into_trend_model();
    let mstl = MSTLModel::new(vec![2], ets);

    // Set up the transformers.
    // These are just illustrative examples; you can use whatever transformers
    // you want.
    let transformers = vec![
        LinearInterpolator::new().boxed(),
        MinMaxScaler::new().boxed(),
        Log::new().boxed(),
    ];

    // Create a forecaster using the transforms.
    let mut forecaster = Forecaster::new(mstl).with_transformers(transformers);

    // Fit the forecaster. This will transform the training data by
    // running the transformers in order, then fit the MSTL model.
    forecaster.fit(DATA).expect("model should fit");

    // Generate some in-sample predictions with 95% prediction intervals.
    // The forecaster will handle back-transforming them onto our original scale.
    let in_sample = forecaster
        .predict_in_sample(0.95)
        .expect("in-sample predictions should work");
    assert_eq!(in_sample.point.len(), DATA.len());
    assert!(in_sample.intervals.is_some());

    // Similarly for out-of-sample predictions:
    let out_of_sample = forecaster
        .predict(5, 0.95)
        .expect("out-of-sample predictions should work");
    assert_eq!(out_of_sample.point.len(), 5);
    assert!(out_of_sample.intervals.is_some());
}
