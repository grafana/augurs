# prophet-wasmstan - a WASM Component for the Prophet Stan model

`prophet-wasmstan` is a WASM component exposing the core model fitting
and sampling functionality of the [Prophet](https://github.com/facebook/prophet)
time series forecasting model. Specifically, this component uses the
generated Stan code (a C++ file) and the Stan library to expose
the `optimize` and `sample` functions of Stan using the Prophet model,
allowing it to be called from a WASM module.

## Building

To build the component you'll need to have several tools from the
WASM Component toolchain installed. The easiest way to do this is
using the `justfile` from the `components` directory of the repository,
which has an `install-dependencies` target that will install all
the necessary tools.

```bash
just install-dependencies
```

Once the dependencies are installed, you can build the component
with the `build-lib-component` target:

```bash
just build
```

This will generate a `prophet-wasmstan.wasm` file in the `prophet-wasmstan`
directory. This file can be used as a WASM component.

## Using the component

The interface exposed by the component is defined in the `prophet-wasmstan.wit`
file.

See the [Component Model docs](https://component-model.bytecodealliance.org/language-support.html)
for instructions on how to use the component in a WASM component in other
languages.

### Javascript

Run the following command to generate Javascript bindings for the component:

```bash
just transpile
```

You should now have Javascript bindings in the `js/prophet-wasmstan` directory.
