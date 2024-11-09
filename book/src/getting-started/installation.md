# Installation

## Rust

Add `augurs` to your `Cargo.toml`. The library is modular, so you only need to enable the features you plan to use:

```toml
[dependencies]
augurs = { version = "0.6.0", features = [] }
```

Available features include:

- `forecaster` - High-level forecasting API with data transformations
- `ets` - Exponential smoothing models
- `mstl` - Multiple Seasonal-Trend decomposition using LOESS
- `outlier` - Outlier detection algorithms
- `clustering` - Time series clustering algorithms
- `dtw` - Dynamic Time Warping distance calculations
- `full` - All features
- `prophet` - Facebook Prophet forecasting model
- `prophet-cmdstan` - Prophet with cmdstan backend
- `prophet-wasmstan` - Prophet with WebAssembly stan backend
- `seasons` - Seasonality detection

For example, to use forecasting with ETS and MSTL:

```toml
[dependencies]
augurs = { version = "0.6.0", features = ["forecaster", "ets", "mstl"] }
```

## Python

The Python bindings can be installed via pip:

```bash
pip install augurs
```

## JavaScript Installation

The JavaScript bindings are available through npm:

```bash
npm install @bsull/augurs
```
