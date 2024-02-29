# High level forecasting API for the augurs time series library

`augurs-forecaster` contains a high-level API for training and predicting with time series models. It currently allows you to combine a model with a set of transformations (such as imputation of missing data, min-max scaling, and log/logit transforms) and fit the model on the transformed data, automatically handling back-transformation of forecasts and prediction intervals.

## Usage

First add this crate, `augurs-core`, and any required model crates to your `Cargo.toml`:

```toml
[dependencies]
augurs-ets = { version = "*", features = ["mstl"] }
augurs-forecaster = "*"
augurs-mstl = "*"
```

```rust
use augurs_ets::{AutoETS, trend::AutoETSTrendModel};
use augurs_forecaster::{Forecaster, Transform, transforms::MinMaxScaleParams};
use augurs_mstl::MSTLModel;

let data = &[
    1.0, 1.2, 1.4, 1.5, f64::NAN, 1.4, 1.2, 1.5, 1.6, 2.0, 1.9, 1.8
];

// Set up the model. We're going to use an MSTL model to handle
// multiple seasonalities, with a non-seasonal `AutoETS` model
// for the trend component.
// We could also use any model that implements `augurs_core::Fit`.
let ets = AutoETS::non_seasonal().into_trend_model();
let mstl = MSTLModel::new(vec![2], ets);

// Set up the transforms.
let transforms = vec![
    Transform::linear_interpolator(),
    Transform::min_max_scaler(MinMaxScaleParams::from_data(data.iter().copied())),
    Transform::log(),
];

// Create a forecaster using the transforms.
let mut forecaster = Forecaster::new(mstl).with_transforms(transforms);

// Fit the forecaster. This will transform the training data by
// running the transforms in order, then fit the MSTL model.
forecaster.fit(&data).expect("model should fit");

// Generate some in-sample predictions with 95% prediction intervals.
// The forecaster will handle back-transforming them onto our original scale.
let in_sample = forecaster
    .predict_in_sample(0.95)
    .expect("in-sample predictions should work");

// Similarly for out-of-sample predictions:
let out_of_sample = forecaster
    .predict(5, 0.95)
    .expect("out-of-sample predictions should work");
```
