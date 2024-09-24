# Changepoint detection models.

Changepoint detection of time series.

For now it is mostly just a wrapper around the [`changepoint`] crate, with
a common `Detector` trait to allow for more implementations in future.

## Example

```rust
use augurs_changepoint::{Detector, DefaultArgpcpDetector};

let data = [0.5, 1.0, 0.4, 0.8, 1.5, 0.9, 0.6, 25.3, 20.4, 27.3, 30.0];
let changepoints = DefaultArgpcpDetector::default().detect_changepoints(&data);
// 0 is always included. 6 is the index prior to the changepoint.
assert_eq!(changepoints, vec![0, 6]);
```

## Credits

The bulk of the actual work is done by the [`changepoint`][changepoint] crate.

[changepoint]: https://crates.io/crates/changepoint
