# Forecasting with Prophet

This tutorial will guide you through using Facebook's Prophet forecasting model with the WebAssembly-based Stan backend in `augurs`. Prophet is particularly well-suited for time series that have strong seasonal effects and multiple seasons.

> **Note**: JavaScript Prophet examples require Node.js 20+ due to ES module requirements in the wasmstan optimizer.

## Prerequisites

<!-- langtabs-start -->
```toml
// Add to your Cargo.toml:
[dependencies]
augurs = { version = "0.6.0", features = ["prophet", "prophet-wasmstan"] }
```

```bash
# Install via npm:
npm install @bsull/augurs

# You'll also need the Prophet WASM Stan optimizer:
npm install @bsull/augurs-prophet-wasmstan
```
<!-- langtabs-end -->

## Basic Prophet Forecasting

Let's start with a simple example:

<!-- langtabs-start -->
```rust
use augurs::prophet::{Prophet, TrainingData, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create timestamps (as Unix timestamps - 20 days of data)
    let start = 1704067200; // 2024-01-01
    let day = 86400; // seconds in a day
    let timestamps: Vec<_> = (0..20).map(|i| start + i * day).collect();

    // Your observations (20 data points)
    let values: Vec<_> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();

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

```javascript
import initProphet, { Prophet } from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

// Initialize WASM modules
await initProphet();

// Create timestamps (as Unix timestamps in seconds - 20 days of data)
const start = 1704067200; // 2024-01-01
const day = 86400; // seconds in a day
const ds = Array.from({ length: 20 }, (_, i) => start + i * day);

// Your observations (20 data points)
const y = Array.from({ length: 20 }, (_, i) => i * 0.5 + 1.0);

// Create and fit Prophet model
const prophet = new Prophet({ optimizer });
prophet.fit({ ds, y });

// Make predictions
const predictions = prophet.predict();

console.log("Predictions:", predictions.yhat.point);
console.log("Lower bounds:", predictions.yhat.lower);
console.log("Upper bounds:", predictions.yhat.upper);
```
<!-- langtabs-end -->

## Adding Regressors

Prophet allows you to include additional regressors to improve your forecasts:

<!-- langtabs-start -->
```rust
use std::collections::HashMap;
use augurs::prophet::{Prophet, TrainingData, Regressor, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create timestamps and values (20 days of data)
    let start = 1704067200;
    let day = 86400;
    let timestamps: Vec<_> = (0..20).map(|i| start + i * day).collect();
    let values: Vec<_> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();

    // Create regressors (20 values to match data length)
    let regressors = HashMap::from([
        (
            "temperature".to_string(),
            (0..20).map(|i| 20.0 + (i as f64) * 0.3).collect::<Vec<_>>(),
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


```javascript
import initProphet, { Prophet } from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

await initProphet();

const start = 1704067200;
const day = 86400;
const ds = Array.from({ length: 20 }, (_, i) => start + i * day);
const y = Array.from({ length: 20 }, (_, i) => i * 0.5 + 1.0);

// Create regressors (additional features - 20 values)
const regressors = {
    temperature: Array.from({ length: 20 }, (_, i) => 20.0 + i * 0.3),
    promotion: Array.from({ length: 20 }, (_, i) => i % 3 === 0 ? 1.0 : 0.0),
};

// Create Prophet with regressors
const prophet = new Prophet({ optimizer });
prophet.fit({ ds, y, regressors });

// Predict with future regressors
// Note: regressors should be provided for each future timestamp
const futurePredictions = prophet.predict({
    ds: Array.from({ length: 5 }, (_, i) => start + (20 + i) * day),
    regressors: {
        temperature: Array.from({ length: 5 }, (_, i) => 20.0 + (20 + i) * 0.3),
        promotion: Array.from({ length: 5 }, (_, i) => (20 + i) % 3 === 0 ? 1.0 : 0.0),
    }
});
console.log("Predictions with regressors:", futurePredictions.yhat.point);
```
<!-- langtabs-end -->

## Customizing the Model

You can customize Prophet's behavior with various options:

<!-- langtabs-start -->
```rust
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

```javascript
import initProphet, { Prophet } from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

await initProphet();

// Historical data (20 days)
const start = 1704067200;
const day = 86400;
const ds = Array.from({ length: 20 }, (_, i) => start + i * day);
const y = Array.from({ length: 20 }, (_, i) => i * 0.5 + 1.0);

// Fit the model
const prophet = new Prophet({ optimizer });
prophet.fit({ ds, y });

// Create future timestamps (3 days ahead)
const futureDates = [
    1704412800, // Day 4
    1704499200, // Day 5
    1704585600, // Day 6
];

// Make predictions for future dates
const predictions = prophet.predict({ ds: futureDates });

console.log("Future predictions:", predictions.yhat.point);
console.log("Uncertainty intervals:", {
    lower: predictions.yhat.lower,
    upper: predictions.yhat.upper,
});
```
<!-- langtabs-end -->

```javascript
import initProphet, { Prophet } from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

await initProphet();

const start = 1704067200;
const day = 86400;
const ds = Array.from({ length: 20 }, (_, i) => start + i * day);
const y = Array.from({ length: 20 }, (_, i) => i * 0.5 + 1.0);

// Configure Prophet with custom settings
const options = {
    // Set growth model
    growth: 'linear', // or 'logistic' or 'flat'
    
    // Configure seasonality
    seasonalityMode: 'multiplicative', // or 'additive'
    yearlySeasonality: { type: "manual", enabled: true },
    weeklySeasonality: { type: "manual", enabled: true },
    dailySeasonality: { type: "manual", enabled: false },
    
    // Adjust changepoint detection
    changepointPriorScale: 0.05,
    changepointRange: 0.8,
    
    // Set uncertainty intervals
    intervalWidth: 0.95,
};

// Create and fit Prophet model with custom options
const prophet = new Prophet({ optimizer, ...options });
prophet.fit({ ds, y });

const predictions = prophet.predict();
console.log("Custom model predictions:", predictions.yhat.point);
```
<!-- langtabs-end -->

## Working with Future Dates

For forecasting into the future, you need to provide future timestamps:

<!-- langtabs-start -->
```rust
use augurs::prophet::{Prophet, TrainingData, PredictionData, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Historical data (20 days)
    let start = 1704067200;
    let day = 86400;
    let timestamps: Vec<_> = (0..20).map(|i| start + i * day).collect();
    let values: Vec<_> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();
    
    let data = TrainingData::new(timestamps, values)?;
    
    // Fit the model
    let optimizer = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(Default::default(), optimizer);
    prophet.fit(data, Default::default())?;

    // Create future timestamps
    let prediction_data = PredictionData::new(vec![
        1704412800, // Day 4
        1704499200, // Day 5
        1704585600, // Day 6
    ]);
    
    let predictions = prophet.predict(Some(prediction_data))?;

    // Access the forecasted values and their bounds
    println!("Predictions: {:?}", predictions.yhat.point);
    println!("Lower bounds: {:?}", predictions.yhat.lower.as_ref().unwrap());
    println!("Upper bounds: {:?}", predictions.yhat.upper.as_ref().unwrap());

    Ok(())
}
```

```javascript
import initProphet, { Prophet } from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

await initProphet();

// Historical data (20 days)
const start = 1704067200;
const day = 86400;
const ds = Array.from({ length: 20 }, (_, i) => start + i * day);
const y = Array.from({ length: 20 }, (_, i) => i * 0.5 + 1.0);

// Fit the model
const prophet = new Prophet({ optimizer });
prophet.fit({ ds, y });

// Create future timestamps (3 days ahead)
const futureDates = [
    1704412800, // Day 4
    1704499200, // Day 5
    1704585600, // Day 6
];

// Make predictions for future dates
const predictions = prophet.predict({ ds: futureDates });

console.log("Future predictions:", predictions.yhat.point);
console.log("Uncertainty intervals:", {
    lower: predictions.yhat.lower,
    upper: predictions.yhat.upper,
});
```
<!-- langtabs-end -->

## Best Practices

### 1. Data Preparation
   - Ensure your timestamps are Unix timestamps (seconds since epoch)
   - Handle missing values before passing to Prophet
   - Consider scaling your target variable if values are very large
   - Sort your data by timestamp

### 2. Model Configuration
   - Start with default settings and adjust based on your needs
   - Use **additive seasonality** for constant seasonal variations
   - Use **multiplicative seasonality** when variations scale with the trend
   - Add regressors for known external factors (holidays, promotions, etc.)

### 3. Performance Considerations
   - **WASMSTAN** runs inside a WASM runtime and may be slower than native code
   - **For server-side Rust**: Consider using the `prophet-cmdstan` feature for better performance
   - **For browsers/Node.js**: WASMSTAN is your best option
   - Large datasets (>1000 points) may require significant computation time
   - The WebAssembly optimizer is ideal for client-side applications and serverless functions

### 4. Language-Specific Notes

**Rust**:
- Full API access with compile-time safety
- Can choose between WASMSTAN and cmdstan backends
- Best for production server applications

**JavaScript**:
- WASM-based Prophet works in both browser and Node.js
- Requires async initialization with `await initProphet()`
- Great for interactive dashboards and client-side analytics
- Note: WASM files need to be served correctly in web applications

## Common Issues and Solutions

### Issue: Predictions Don't Match Expected Seasonality

**Solution**: Adjust seasonality settings and ensure enough historical data

<!-- langtabs-start -->
```rust
use augurs::prophet::{Prophet, TrainingData, ProphetOptions, SeasonalityOption, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ProphetOptions {
        yearly_seasonality: SeasonalityOption::Manual(true),
        weekly_seasonality: SeasonalityOption::Manual(true),
        daily_seasonality: SeasonalityOption::Auto, // Let Prophet decide
        ..Default::default()
    };
    
    let optimizer = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(options, optimizer);
    
    let start = 1704067200;
    let day = 86400;
    let timestamps: Vec<_> = (0..20).map(|i| start + i * day).collect();
    let values: Vec<_> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let data = TrainingData::new(timestamps, values)?;
    
    prophet.fit(data, Default::default())?;
    println!("Model fitted with custom seasonality!");
    
    Ok(())
}
```

```javascript
import initProphet, { Prophet } from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

await initProphet();

const options = {
    yearlySeasonality: { type: "manual", enabled: true },
    weeklySeasonality: { type: "manual", enabled: true },
    dailySeasonality: { type: "auto" },
};

const prophet = new Prophet({ optimizer, ...options });

const start = 1704067200;
const day = 86400;
const ds = Array.from({ length: 20 }, (_, i) => start + i * day);
const y = Array.from({ length: 20 }, (_, i) => i * 0.5 + 1.0);

// Fit model with custom options
prophet.fit({ ds, y });
console.log("Model fitted with custom seasonality!");
```
<!-- langtabs-end -->

### Issue: Model Takes Too Long to Fit

**Solution**: Reduce the number of changepoints or use a simpler growth model

<!-- langtabs-start -->
```rust
use augurs::prophet::{Prophet, TrainingData, ProphetOptions, wasmstan::WasmstanOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ProphetOptions {
        n_changepoints: 10, // Reduce from default 25
        changepoint_range: 0.8.try_into().unwrap(),
        ..Default::default()
    };
    
    let optimizer = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(options, optimizer);
    
    let start = 1704067200;
    let day = 86400;
    let timestamps: Vec<_> = (0..20).map(|i| start + i * day).collect();
    let values: Vec<_> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let data = TrainingData::new(timestamps, values)?;
    
    prophet.fit(data, Default::default())?;
    println!("Model fitted with custom options!");
    
    Ok(())
}
```

```javascript
import { Prophet } from '@bsull/augurs/prophet';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

const options = {
    nChangepoints: 10, // Reduce from default 25
    changepointRange: 0.8,
};

const prophet = new Prophet({ optimizer, ...options });

const start = 1704067200;
const day = 86400;
const ds = Array.from({ length: 20 }, (_, i) => start + i * day);
const y = Array.from({ length: 20 }, (_, i) => i * 0.5 + 1.0);

prophet.fit({ ds, y });
console.log("Model fitted with custom options!");
```
<!-- langtabs-end -->

## Next Steps

- Explore [MSTL forecasting](../getting-started/quick-start.md#basic-forecasting) for an alternative decomposition-based approach
- Learn about [outlier detection](./outlier-detection.md) to clean your data before forecasting
- Check the [API documentation](../api/index.md) for advanced Prophet features
- For production use in Rust, consider the `prophet-cmdstan` feature for better performance

This comprehensive guide demonstrates how to use Prophet for time series forecasting in both Rust and JavaScript, with practical examples and best practices for real-world applications.

## Troubleshooting

Common issues and their solutions:

- **Invalid timestamps**: Ensure timestamps are Unix timestamps in seconds
- **Missing values**: Prophet can handle some missing values, but it's better to preprocess them
- **Convergence issues**: Try adjusting the number of iterations or sampling parameters

## Next Steps

- Learn about [changepoint detection](../how-to/changepoints.md)
- Explore [seasonal decomposition](../how-to/seasonality.md)
- Understand [cross-validation](../how-to/cross-validation.md)
