//! Example of using the Prophet model with the cmdstan optimizer.
//!
//! To run this example, you must first download the Prophet Stan model
//! and libtbb shared library into the `prophet_stan_model` directory.
//! The easiest way to do this is to run the `download-stan-model`
//! binary in the `augurs-prophet` crate:
//!
//! ```sh
//! $ cargo run --features download --bin download-stan-model
//! $ cargo run --example prophet_cmdstan
//! ```
use std::{collections::HashMap, time::Duration};

use augurs::prophet::{cmdstan::CmdstanOptimizer, Prophet, Regressor, TrainingData};

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
    let data = TrainingData::new(ds, y.clone())?
        .with_regressors(HashMap::from([
            (
                "foo".to_string(),
                vec![
                    1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
                ],
            ),
            (
                "bar".to_string(),
                vec![
                    4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0,
                ],
            ),
        ]))
        .unwrap();

    let cmdstan = CmdstanOptimizer::with_prophet_path("prophet_stan_model/prophet_model.bin")?
        .with_poll_interval(Duration::from_millis(100))
        .with_refresh(50);
    // If you were using the embedded version of the cmdstan model, you'd use this:
    // let cmdstan = CmdstanOptimizer::new_embedded();

    let mut prophet = Prophet::new(Default::default(), cmdstan);
    prophet.add_regressor("foo".to_string(), Regressor::additive());
    prophet.add_regressor("bar".to_string(), Regressor::additive());

    prophet.fit(data, Default::default())?;
    let predictions = prophet.predict(None)?;
    assert_eq!(predictions.yhat.point.len(), y.len());
    assert!(predictions.yhat.lower.is_some());
    assert!(predictions.yhat.upper.is_some());
    println!("Predicted values: {:#?}", predictions.yhat);
    Ok(())
}
