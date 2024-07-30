# augurs: time series forecasting

Javascript bindings to the [`augurs`][repo] time series framework.

## Usage

1. Add the package to your dependencies:

```json
"dependencies": {
    "@bsull/augurs": "^0.3.0"
}
```

1. Import the default function and initialize once somewhere in your application:

```javascript
import init from "@bsull/augurs";
init().then(() => console.log("Initialized augurs"));
```

1. Use the various ETS, changepoint, outlier, or seasonality detection algorithms. For example:

```javascript
import { ets, seasonalities } from "@bsull/augurs"

const y = new Float64Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]); // your time series data
const seasonLengths = seasonalities(y);
const model = ets(seasonLengths, { impute: true });
model.fit(y);

const predictionInterval = 0.95;
// Generate in-sample predictions for the training set.
const { point, lower, upper } = model.predictInSample(predictionInterval);
// Generate out-of-sample forecasts.
const { point: futurePoint, lower: futureLower, upper: futureUpper } = model.predict(10, predictionInterval);
```

[repo]: https://github.com/grafana/augurs
