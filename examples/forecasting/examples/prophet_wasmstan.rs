//! Example of using the Prophet model with the wasmstan optimizer.
use std::collections::HashMap;

use augurs::prophet::{wasmstan::WasmstanOptimizer, Prophet, Regressor, TrainingData};

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

    let wasmstan = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(Default::default(), wasmstan);
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
