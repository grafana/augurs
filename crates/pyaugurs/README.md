# Python bindings to the augurs time series framework

## Installation

Eventually wheels will be provided as part of GitHub releases and maybe even on PyPI.
At that point it will be as easy as:

```shell
$ pip install augurs
```

Until then it's a bit more manual. You'll need [maturin] installed and a local copy of this
repository. Then, from the `crates/pyaugurs` directory, with your virtualenv activated:

```shell
$ maturin build --release
```

You'll probably want numpy as well:

```shell
$ pip install numpy
```

## Usage

### Multiple Seasonal Trend Decomposition with LOESS (MSTL) models

```python
import augurs as aug
import numpy as np

y = np.array([1.5, 3.0, 2.5, 4.2, 2.7, 1.9, 1.0, 1.2, 0.8])
periods = [3, 4]
# Use an AutoETS trend forecaster
model = aug.MSTL.ets(periods)
model.fit(y)
out_of_sample = model.predict(10, level=0.95)
print(out_of_sample.point())
print(out_of_sample.lower())
in_sample = model.predict_in_sample(level=0.95)

# Or use your own forecaster
class CustomForecaster:
    """See docs for more details on how to implement this."""    
    def fit(self, y: np.ndarray):
        pass
    def predict(self, horizon: int, level: float | None) -> aug.Forecast:
        return aug.Forecast(point=np.array([5.0, 6.0, 7.0]))
    def predict_in_sample(self, level: float | None) -> aug.Forecast:
        return aug.Forecast(point=y)
    ...

model = aug.MSTL.custom_trend(periods, aug.TrendModel(CustomForecaster()))
model.fit(y)
model.predict(10, level=0.95)
model.predict_in_sample(level=0.95)
```

### Exponential smoothing models

```python
import augurs as aug
import numpy as np

y = np.array([1.5, 3.0, 2.5, 4.2, 2.7, 1.9, 1.0, 1.2, 0.8])
model = aug.AutoETS(3, "ZZN")
model.fit(y)
model.predict(10, level=0.95)
```

More to come!

[maturin]: https://www.maturin.rs/
