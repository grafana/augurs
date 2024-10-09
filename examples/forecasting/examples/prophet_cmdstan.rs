use augurs::prophet::{cmdstan::CmdstanOptimizer, Prophet, TrainingData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ds = vec![
        1704067200, 1704871384, 1705675569, 1706479753, 1707283938, 1708088123, 1708892307,
        1709696492, 1710500676, 1711304861, 1712109046, 1712913230, 1713717415,
    ];
    let y = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let data = TrainingData::new(ds, y.clone())?;
    let mut prophet = Prophet::new(Default::default(), CmdstanOptimizer::new());
    prophet.fit(data, Default::default())?;
    let predictions = prophet.predict(None)?;
    assert_eq!(predictions.yhat.point.len(), y.len());
    assert!(predictions.yhat.lower.is_some());
    assert!(predictions.yhat.upper.is_some());
    println!("Predicted values: {:#?}", predictions.yhat);
    Ok(())
}
