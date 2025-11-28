# Forecasting with Prophet

This tutorial will guide you through using Facebook's Prophet forecasting model with the WebAssembly-based Stan backend in `augurs`. Prophet is particularly well-suited for time series that have strong seasonal effects and multiple seasons.

## Prerequisites

First, add the necessary features to your `Cargo.toml`:

```toml
[dependencies]
augurs = { version = "0.6.0", features = ["prophet", "prophet-wasmstan"] }
```

## Basic Prophet Forecasting

Let's start with a simple example:

```rust,no_run
# extern crate augurs;
use augurs::prophet::{Prophet, TrainingData, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create timestamps (as Unix timestamps)
    let timestamps = vec![
        1704067200, // 2024-01-01
        1704153600, // 2024-01-02
        1704240000, // 2024-01-03
        1704326400, // 2024-01-04
        1704412800, // 2024-01-05
        // ... more dates
    ];

    // Your observations
    let values = vec![1.1, 2.1, 3.2, 4.3, 5.5];

    // Create training data
    let data = TrainingData::new(timestamps, values)?;

    // Initialize Prophet with WASMSTAN optimizer
    let optimizer = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(Default::default(), optimizer);

    // Fit the model
    prophet.fit(data, Default::default())?;

    // Make in-sample predictions
    let predictions = prophet.predict(None)?;

    println!("Predictions: {:?}", predictions.yhat.point);
    println!("Lower bounds: {:?}", predictions.yhat.lower.unwrap());
    println!("Upper bounds: {:?}", predictions.yhat.upper.unwrap());

    Ok(())
}
```

## Adding Regressors

Prophet allows you to include additional regressors to improve your forecasts:

```rust,no_run
# extern crate augurs;
use std::collections::HashMap;
use augurs::prophet::{Prophet, TrainingData, Regressor, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create timestamps and values as before
    let timestamps = vec![
        1704067200, // 2024-01-01
        1704153600, // 2024-01-02
        1704240000, // 2024-01-03
        1704326400, // 2024-01-04
        1704412800, // 2024-01-05
    ];
    let values = vec![1.1, 2.1, 3.2, 4.3, 5.5];

    // Create regressors
    let regressors = HashMap::from([
        (
            "temperature".to_string(),
            vec![20.0, 22.0, 21.0, 21.5, 22.5], // temperature values
        ),
    ]);

    // Create training data with regressors
    let data = TrainingData::new(timestamps, values)?
        .with_regressors(regressors)?;

    // Initialize Prophet
    let optimizer = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(Default::default(), optimizer);

    // Add regressors with their modes
    prophet.add_regressor("temperature".to_string(), Regressor::additive());

    // Fit and predict as before
    prophet.fit(data, Default::default())?;
    let predictions = prophet.predict(None)?;

    Ok(())
}
```

## Customizing the Model

Prophet offers several customization options:

```rust,ignore
# extern crate augurs;
use augurs::prophet::{
    Prophet, TrainingData, ProphetOptions, FeatureMode,
    GrowthType, SeasonalityOption,
    wasmstan::WasmstanOptimizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure Prophet with custom settings
    let options = ProphetOptions {
        // Set growth model
        growth: GrowthType::Linear,
        // Configure seasonality
        seasonality_mode: FeatureMode::Multiplicative,
        yearly_seasonality: SeasonalityOption::Manual(true),
        weekly_seasonality: SeasonalityOption::Manual(true),
        daily_seasonality: SeasonalityOption::Manual(false),
        ..Default::default()
    };

    let optimizer = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(options, optimizer);

    // Proceed with fitting and prediction...

    Ok(())
}
```

## Working with Future Dates

To forecast into the future, you'll need to create a `PredictionData` object with the timestamps you want
to predict. It must also contain the same regressors as the training data:

```rust,no_run
# extern crate augurs;
use augurs::prophet::{Prophet, PredictionData, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup and fit model as before...
    # let optimizer = WasmstanOptimizer::new();
    # let mut prophet = Prophet::new(Default::default(), optimizer);

    let prediction_data = PredictionData::new(vec![
        1704499200, // 2024-01-06
        1704585600, // 2024-01-07
    ]);
    let predictions = prophet.predict(Some(prediction_data))?;

    // Access the forecasted values, and their bounds.
    println!("Predictions: {:?}", predictions.yhat.point);
    println!("Lower bounds: {:?}", predictions.yhat.lower.as_ref().unwrap());
    println!("Upper bounds: {:?}", predictions.yhat.upper.as_ref().unwrap());

    Ok(())
}
```

## Best Practices

1. **Data Preparation**
   - Ensure your timestamps are Unix timestamps
   - Handle missing values before passing to Prophet
   - Consider scaling your target variable if values are very large

2. **Model Configuration**
   - Start with default settings and adjust based on your needs
   - Use additive seasonality for constant seasonal variations
   - Use multiplicative seasonality when variations scale with the trend

3. **Performance Considerations**
   - WASMSTAN runs inside a WASM runtime and may be slower than native code
   - For server-side applications, consider using the `prophet-cmdstan` feature instead
   - Large datasets may require more computation time

## Troubleshooting

Common issues and their solutions:

- **Invalid timestamps**: Ensure timestamps are Unix timestamps in seconds
- **Missing values**: Prophet can handle some missing values, but it's better to preprocess them
- **Convergence issues**: Try adjusting the number of iterations or sampling parameters

## Next Steps

- Learn about [changepoint detection](../how-to/changepoints.md)
- Explore [seasonal decomposition](../how-to/seasonality.md)
- Understand [cross-validation](../how-to/cross-validation.md)
