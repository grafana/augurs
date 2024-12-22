//! Example of using the Prophet model with the wasmstan optimizer.

use augurs::{
    forecaster::{transforms::MinMaxScaler, Forecaster, Transformer},
    prophet::{wasmstan::WasmstanOptimizer, Prophet, TrainingData},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    tracing::info!("Running Prophet example");

    let ds = vec![
        1704067200, 1704871384, 1705675569, 1706479753, 1707283938, 1708088123, 1708892307,
        1709696492, 1710500676, 1711304861, 1712109046, 1712913230, 1713717415,
    ];
    let y = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let data = TrainingData::new(ds, y.clone())?;

    // Set up the transformers.
    // These are just illustrative examples; you can use whatever transformers
    // you want.
    let transformers = vec![MinMaxScaler::new().boxed()];

    // Set up the model. Create the Prophet model as normal, then convert it to a
    // `ProphetForecaster`.
    let prophet = Prophet::new(Default::default(), WasmstanOptimizer::new());
    let prophet_forecaster = prophet.into_forecaster(data.clone(), Default::default());

    // Finally create a Forecaster using those transforms.
    let mut forecaster = Forecaster::new(prophet_forecaster).with_transformers(transformers);

    // Fit the forecaster. This will transform the training data by
    // running the transformers in order, then fit the Prophet model.
    forecaster.fit(&y).expect("model should fit");

    // Generate some in-sample predictions with 95% prediction intervals.
    // The forecaster will handle back-transforming them onto our original scale.
    let predictions = forecaster.predict_in_sample(0.95)?;
    assert_eq!(predictions.point.len(), y.len());
    assert!(predictions.intervals.is_some());
    println!("In-sample predictions: {:?}", predictions);

    // Generate 10 out-of-sample predictions with 95% prediction intervals.
    let predictions = forecaster.predict(10, 0.95)?;
    assert_eq!(predictions.point.len(), 10);
    assert!(predictions.intervals.is_some());
    println!("Out-of-sample predictions: {:?}", predictions);
    Ok(())
}
