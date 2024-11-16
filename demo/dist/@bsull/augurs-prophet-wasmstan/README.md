# Prophet Stan model, compiled to WASM

This is a WASM-compiled version of the [Prophet](https://facebook.github.io/prophet/) Stan model, for use with the [@bsull/augurs](https://github.com/grafana/augurs) library.

## Usage

```js
import { Prophet } from '@bsull/augurs';
import { optimizer } from '@bsull/augurs-prophet-wasmstan';

// Create some fake data.
// `ds` must be timestamps since the epoch, in seconds.
const ds = [1704067200, 1704871384, 1705675569, 1706479753, 1707283938, 1708088123,
  1708892307, 1709696492, 1710500676, 1711304861, 1712109046, 1712913230,
];
const y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
const trainingData = { ds, y };

// Create a Prophet model and fit it to the training data.
const prophet = new Prophet();
prophet.fit(trainingdata);
// Predict for the training set.
prophet.predict();
// Predict for a new time point.
prophet.predict({ ds: [ 1713717414 ]})
```

See the documentation for `@bsull/augurs` for more details.

## Troubleshooting

### Webpack

The generated Javascript bindings in this package may require some additional Webpack configuration to work.
Adding this to your `webpack.config.js` should be enough:

```javascript
{
  experiments: {
    // Required to load WASM modules.
    asyncWebAssembly: true,
  },
  resolve: {
    fallback: {
      fs: false,
    },
  },
}
```
