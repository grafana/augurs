# Seasonality detection for time series

`augurs-seasons` contains methods for detecting seasonality or periodicity in time series.

It currently contains implementations to do so using periodograms, similar to the [`seasonal`] Python package.

## Usage

```rust
use augurs_seasons::{Detector, PeriodogramDetector};

# fn main() {
let y = &[
    0.1, 0.3, 0.8, 0.5,
    0.1, 0.31, 0.79, 0.48,
    0.09, 0.29, 0.81, 0.49,
    0.11, 0.28, 0.78, 0.53,
    0.1, 0.3, 0.8, 0.5,
    0.1, 0.31, 0.79, 0.48,
    0.09, 0.29, 0.81, 0.49,
    0.11, 0.28, 0.78, 0.53,
];
// Use the detector with default parameters.
let periods = PeriodogramDetector::default().detect(y);
assert_eq!(periods[0], 4);

// Customise the detector using the builder.
let periods = PeriodogramDetector::builder()
    .min_period(4)
    .max_period(8)
    .threshold(0.8)
    .build()
    .detect(y);
assert_eq!(periods[0], 4);
# }
```

## Credits

This implementation is based heavily on the [`seasonal`] Python package.
It also makes heavy use of the [`welch-sde`] crate.

[`seasonal`]: https://github.com/welch/seasonal
[`welch-sde`]: https://crates.io/crates/welch-sde

## License

Dual-licensed to be compatible with the Rust project.
Licensed under the Apache License, Version 2.0 `<http://www.apache.org/licenses/LICENSE-2.0>` or the MIT license `<http://opensource.org/licenses/MIT>`, at your option.
