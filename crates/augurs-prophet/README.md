# Prophet: forecasting at scale

`augurs-prophet` contains an implementation of the [Prophet]
time series forecasting library.

**Caveats**

This crate has been tested fairly thoroughly but Prophet contains a lot of options - please [report any bugs][bugs] you find!

Currently, only MLE/MAP based optimization is supported. This means that uncertainty in seasonality components isn't modeled.
Contributions to add sampling capabilities are welcome - please get in touch in the
[issue tracker][feature request] if you're interested in this.

## Example (WASM-compiled Stan)

First enable the `wasmstan` feature of this crate. Then:

```rust
use augurs::prophet::{wasmstan::WasmstanOptimizer, Prophet, TrainingData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ds = vec![
        1704067200, 1704871384, 1705675569, 1706479753, 1707283938, 1708088123, 1708892307,
        1709696492, 1710500676, 1711304861, 1712109046, 1712913230, 1713717415,
    ];
    let y = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let data = TrainingData::new(ds, y.clone())?;

    let optimizer = WasmstanOptimizer::new();
    let mut prophet = Prophet::new(Default::default(), optimizer);

    prophet.fit(data, Default::default())?;
    let predictions = prophet.predict(None)?;
    assert_eq!(predictions.yhat.point.len(), y.len());
    assert!(predictions.yhat.lower.is_some());
    assert!(predictions.yhat.upper.is_some());
    println!("Predicted values: {:#?}", predictions.yhat);
    Ok(())
}
```

## Example (cmdstan)

First, download the Prophet Stan model using the included binary:

```sh
$ cargo install --bin download-stan-model --features download augurs-prophet
$ download-stan-model
Downloading https://files.pythonhosted.org/packages/1f/47/f7d10a904756830efd8522700e582822ff44a15f839b464044ee4c53ee36/prophet-1.1.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl to prophet_stan_model/prophet-1.1.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Writing zipped prophet/stan_model/prophet_model.bin to prophet_stan_model/prophet_model.bin
Writing zipped prophet.libs/libtbb-dc01d64d.so.2 to prophet_stan_model/lib/libtbb-dc01d64d.so.2
```

Then enable the `cmdstan` feature of this crate and use the `Prophet` model as follows:

```rust,no_run
use augurs::prophet::{cmdstan::CmdstanOptimizer, Prophet, TrainingData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ds = vec![
        1704067200, 1704871384, 1705675569, 1706479753, 1707283938, 1708088123, 1708892307,
        1709696492, 1710500676, 1711304861, 1712109046, 1712913230, 1713717415,
    ];
    let y = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let data = TrainingData::new(ds, y.clone())?;

    let optimizer = CmdstanOptimizer::with_prophet_path("prophet_stan_model/prophet_model.bin")?;
    // If you were using the embedded version of the cmdstan model, you'd enable
    // the `compile-cmdstan` feature and use this:
    //
    // let optimizer = CmdstanOptimizer::new_embedded();

    let mut prophet = Prophet::new(Default::default(), optimizer);

    prophet.fit(data, Default::default())?;
    let predictions = prophet.predict(None)?;
    assert_eq!(predictions.yhat.point.len(), y.len());
    assert!(predictions.yhat.lower.is_some());
    assert!(predictions.yhat.upper.is_some());
    println!("Predicted values: {:#?}", predictions.yhat);
    Ok(())
}
```

Note that the `CmdstanOptimizer` needs to know the path to the Prophet
model binary.

This crate aims to be low-dependency to enable it to run in as
many places as possible. With that said, we need to talk about
optimizers…

## Optimizers

The original Prophet library uses [Stan] to handle optimization and MCMC sampling.
Stan is a platform for statistical modeling which can perform Bayesian statistical
inference as well as maximum likelihood estimation using optimizers such as L-BFGS.
However, it is written in C++ and has non-trivial dependencies, which makes it
difficult to interface with from Rust (or, indeed, Python).

Similar to the Python library, `augurs-prophet` abstracts MLE optimization
using the `Optimizer` and (later) MCMC using the `Sampler` traits.
There are several implementations of the `Optimizer` trait, and some
ideas for more, all documented below.

### `wasmstan`

Using WASI and [WASM components], we can compile the Stan model
to WebAssembly. This is done in the [`components/cpp/prophet-wasmstan`][repo-dir]
directory of the `augurs` repository.

The `wasmstan` module of this crate makes use of this WASM component
and provides an `Optimizer` which runs it inside a Wasmtime runtime.
This ensures we're using all the same Stan code as the initial
implementation, but requiring a Stan installation. It even performs
roughly as well as the native Stan code in release mode.

This also has the advantage that the WASM component can be used in
a browser. The `augurs-js` crate contains a slightly different
`Optimizer` implementation which does this using the browser's
WASM runtime rather than including Wasmtime, to reduce the
bundle size.

For WASM, we could abstract the C++ side of things behind a
[WASM component] which exposes an `optimize` interface,
and create a second Prophet component which imports that
interface to implement the `Optimizer` trait of this crate.

### `cmdstan`

The `cmdstan` module of this crate contains an implementation of `Optimizer`
which will use a compiled Stan program to do this. See the `cmdstan` module
for more details on how to use it.

This requires the `cmdstan` feature to be enabled, and optionally the
`compile-cmdstan` feature to be enabled if you want to compile and embed
the Stan model at build time.

This mimics the approach now taken by the Python implementation, which uses
the `cmdstanpy` package and compiles the Stan program into a standalone
binary on installation. It then executes that binary during the fitting
stage to perform optimization or sampling, passing the data and
parameters between Stan and Python using files on the filesystem.

This works fine if you're operating in a desktop or server environment,
but poses issues when running in more esoteric environments such as
WebAssembly.

### `libstan`

We could choose to write a `libstan` crate which uses [`cxx`][cxx] to
interface directly with the C++ library generated by Stan. Since the
model code is constant (unless we upgrade the version of `stanc` used to
generate it), we could also write a small amount of C++ to make it
possible for us to pass data directly to it from Rust.

In theory this should work OK for any target which Stan can compile to.
The problem I've noticed is that Stan isn't particularly careful about
which headers it imports, so even just compiling the `model.hpp` library,
you end up with a bunch of I/O and filesystem related headers imported,
which aren't available when using standard WASM.

Perhaps we could clean Stan up so it didn't import those things? We should
be able to target most environments in that case.

### A reimplementation of Stan

We could re-implement Stan in a new Rust crate and use that
here. This is likely to be by far the largest amount of work!

## Credits

This implementation is based heavily on the original [Prophet] Python
package. Some changes have been made to make the APIs more idiomatic
Rust or to take advantage of the type system.

[bugs]: https://github.com/grafana/augurs/issues/new?labels=bug%2Cprophet
[feature request]: https://github.com/grafana/augurs/issues/new?labels=enhancement%2Cprophet
[Prophet]: https://facebook.github.io/prophet/
[Stan]: https://mc-stan.org/
[cxx]: https://cxx.rs/
[WASM components]: https://component-model.bytecodealliance.org/
[repo-dir]: https://github.com/grafana/augurs/tree/main/components/cpp/prophet-wasmstan
