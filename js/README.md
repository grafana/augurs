# augurs

[![npm](https://img.shields.io/npm/v/@bsull/augurs)](https://www.npmjs.com/package/@bsull/augurs)
[![npm](https://img.shields.io/npm/dm/@bsull/augurs)](https://www.npmjs.com/package/@bsull/augurs)
[![npm](https://img.shields.io/npm/l/@bsull/augurs)](https://www.npmjs.com/package/@bsull/augurs)

JavaScript bindings for the augurs time series framework.

## Installation

Add this package to your project with:

```bash
npm install @bsull/augurs
```

## Usage

Full usage docs are still to come, but here's a quick example:

```js
import initProphet, { Prophet } from '@bsull/augurs/prophet';
import initTransforms, { Pipeline, Transform } from '@bsull/augurs/transforms';
// Note: you'll need this extra package if you want to use the Prophet model.
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

// Initialize the WASM components before using any augurs functions.
await Promise.all([initProphet(), initTransforms()]);

// Create a pipeline which will apply a Yeo-Johnson transform and a standard scaler.
const pipeline = new Pipeline([
  new Transform('yeoJohnson'),
  new Transform('standardScaler'),
]);

// Create a Prophet model with the WASM-based optimizer.
const prophet = new Prophet({ optimizer });

const ds = [1704067200, 1704871384, 1705675569, 1706479753, 1707283938, 1708088123,
const y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

// Fit the pipeline to the data.
const yTransformed = pipeline.fitTransform(y);

// Fit the Prophet model to the transformed data.
prophet.fit({ ds, y: yTransformed });

// Make in-sample predictions and back-transform them.
const preds = prophet.predict();
const yhat = {
  point: pipeline.inverseTransform(preds.yhat.point),
  intervals: {
    lower: pipeline.inverseTransform(preds.yhat.lower),
    upper: pipeline.inverseTransform(preds.yhat.upper),
  },
};
```

See the [documentation](https://docs.augu.rs/js/getting-started/quick-start) for more information.

## License

This project is dual-licensed under the [Apache 2.0](LICENSE-APACHE) and [MIT](LICENSE-MIT) licenses.
