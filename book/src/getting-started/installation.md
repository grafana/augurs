# Installation

`augurs` is available for Rust, JavaScript, and Python. Choose your preferred language below to get started.

## Quick Install

<!-- langtabs-start -->
```toml
// Add to your Cargo.toml:
[dependencies]
augurs = { version = "0.6.0", features = ["forecaster", "ets", "mstl"] }
```

```bash
# Install via npm:
npm install @bsull/augurs

# Or with yarn:
yarn add @bsull/augurs
```

```bash
# Install via pip:
pip install augurs

# Or with poetry:
poetry add augurs
```
<!-- langtabs-end -->

## Detailed Installation

### Rust

Add `augurs` to your `Cargo.toml`. The library is modular, so you only need to enable the features you plan to use:

```toml
[dependencies]
augurs = { version = "0.6.0", features = [] }
```

#### Available Features

- `forecaster` - High-level forecasting API with data transformations
- `ets` - Exponential smoothing models
- `mstl` - Multiple Seasonal-Trend decomposition using LOESS
- `outlier` - Outlier detection algorithms
- `clustering` - Time series clustering algorithms
- `dtw` - Dynamic Time Warping distance calculations
- `seasons` - Seasonality detection
- `prophet` - Facebook Prophet forecasting model
- `prophet-cmdstan` - Prophet with cmdstan backend
- `prophet-wasmstan` - Prophet with WebAssembly stan backend
- `full` - All features

#### Common Feature Combinations

For forecasting with ETS and MSTL:
```toml
[dependencies]
augurs = { version = "0.6.0", features = ["forecaster", "ets", "mstl"] }
```

For outlier detection and clustering:
```toml
[dependencies]
augurs = { version = "0.6.0", features = ["outlier", "clustering", "dtw"] }
```

For everything:
```toml
[dependencies]
augurs = { version = "0.6.0", features = ["full"] }
```

### JavaScript

The JavaScript bindings are available as an npm package and work in both Node.js and browser environments.

#### Node.js

```bash
npm install @bsull/augurs
```

Then import in your code:
```javascript
import { MSTLModel, AutoETS } from '@bsull/augurs';
```

#### Browser

You can also use augurs in the browser via a CDN or by bundling with your favorite bundler (webpack, vite, rollup, etc.).

```html
<script type="module">
  import { MSTLModel } from 'https://cdn.jsdelivr.net/npm/@bsull/augurs/+esm';
</script>
```

### Python

The Python bindings can be installed via pip:

```bash
pip install augurs
```

Or with poetry:
```bash
poetry add augurs
```

Then import in your Python code:
```python
from augurs import MSTLModel, AutoETS
from augurs.outlier import MADDetector
```

## Verifying Installation

Test your installation with a simple example:

<!-- langtabs-start -->
```rust
use augurs::mstl::MSTLModel;

fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let model = MSTLModel::naive(vec![2]);
    println!("augurs installed successfully!");
}
```

```javascript
import { MSTL } from '@bsull/augurs/mstl';

const data = [1.0, 2.0, 3.0, 4.0, 5.0];
const model = MSTL.ets([2]);
console.log("augurs installed successfully!");
```

```python
import augurs as aug
import numpy as np

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
model = aug.MSTL.ets([2])
print("augurs installed successfully!")
```
<!-- langtabs-end -->

## Requirements

<!-- langtabs-start -->
```bash
# Minimum Rust version: 1.70
# Check your Rust version:
rustc --version
```

```bash
# Requires Node.js 16 or higher
# Check your Node.js version:
node --version
```

```bash
# Requires Python 3.8 or higher
# Check your Python version:
python --version
```
<!-- langtabs-end -->

## Next Steps

Now that you have `augurs` installed, check out the [Quick Start Guide](./quick-start.md) to learn how to use it!