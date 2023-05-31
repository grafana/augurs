# Exponential smoothing models.

This crate provides exponential smoothing models for time series forecasting
in the `augurs` framework. The models are implemented entirely in Rust and are based
on the [statsforecast][] Python package.

**Important**: This crate is still in development and the API is subject to change.
Seasonal models are not yet implemented, and some model types have not been tested.

# Example

```
use augurs_ets::AutoETS;

let data: Vec<_> = (0..10).map(|x| x as f64).collect();
let mut search = AutoETS::new(1, "ZZN")
    .expect("ZZN is a valid model search specification string");
let model = search.fit(&data).expect("fit should succeed");
let forecast = model.predict(5, 0.95);
assert_eq!(forecast.point.len(), 5);
assert_eq!(forecast.point, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
```

## Credits

This implementation is based heavily on the [statsforecast implementation][statsforecast-impl].

## References

- [Rob J. Hyndman, Yeasmin Khandakar (2008). “Automatic Time Series Forecasting: The forecast package for R”.][hyndman-khandakar]
- [Hyndman, Rob, et al (2008). “Forecasting with exponential smoothing: the state space approach”.][hyndman-et-al]

[hyndman-khandakar]: https://www.jstatsoft.org/article/view/v027i03
[hyndman-et-al]: https://robjhyndman.com/expsmooth/
[statsforecast]: https://nixtla.github.io/statsforecast/
[statsforecast-impl]: https://nixtla.github.io/statsforecast/models.html#autoets
