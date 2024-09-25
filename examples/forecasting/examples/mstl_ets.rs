//! This example demonstrates how to use MSTL to forecast the next 10 values
//! of a time series using an ETS model for the trend component.

use augurs::{ets::AutoETS, mstl::MSTLModel, prelude::*};

// Input data must be a `&[f64]` for the MSTL algorithm.
const DATA: &[f64] = &[
    1.0, 1.2, 1.4, 1.5, 1.4, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8, 1.9, 2.0,
];

fn main() {
    // Define the number of seasonal observations per period.
    // In this example we have daily data with weekly seasonality,
    // so our periods are 7 days.
    let periods = vec![7];

    // Create an ETS model for the trend component.
    // Note that this requires both the `ets` and `mstl` features of `augurs`.
    let ets = AutoETS::non_seasonal().into_trend_model();

    // Create an MSTL model using a naive trend forecaster.
    // Note: in real life you may want to use a different
    // trend forecaster.
    let mstl = MSTLModel::new(periods, ets);

    // Fit the model. Note this consumes `mstl` and returns
    // a fitted version.
    let fit = mstl.fit(DATA).expect("model should fit");

    // Obtain in-sample and out-of-sample predictions, along
    // with prediction intervals.
    let in_sample = fit
        .predict_in_sample(0.95)
        .expect("in-sample predictions should work");
    assert_eq!(in_sample.point.len(), DATA.len());
    assert!(in_sample.intervals.is_some());
    let out_of_sample = fit
        .predict(10, 0.95)
        .expect("out-of-sample predictions should work");
    assert_eq!(out_of_sample.point.len(), 10);
    assert!(out_of_sample.intervals.is_some());
}
