//! An optimizer that uses `wasmtime` to run the Prophet model inside a WebAssembly component.
//!
//! This module contains an [`Optimizer`] implementation that uses the Stan model
//! compiled to WebAssembly to optimize the model parameters, by running the model
//! inside Wasmtime.
//!
//! The component itself is embedded in the binary, so no additional files are needed.
//! The code used to create the WASM module can be found in the [augurs repository][repo].
//!
//! To use the optimizer, simply enable the `wasmstan` feature of this crate and
//! create a new `WasmstanOptimizer` instance using [`WasmstanOptimizer::new`].
//! This can be passed as the `optimizer` argument when creating a new
//! [`Prophet`](crate::Prophet) instance.
//!
//! Note that this optimizer runs rather slowly in debug mode, but benchmarks show
//! it to be competitive with `cmdstan` in release mode.
//!
//! [repo]: https://github.com/grafana/augurs/tree/main/components/cpp/prophet-wasmstan

use std::fmt;

use wasmtime::{
    component::{Component, Linker},
    Engine, Store,
};
use wasmtime_wasi::{ResourceTable, WasiCtx, WasiView};

use crate::{
    optimizer::{self, Data, InitialParams, OptimizeOpts, OptimizedParams},
    Optimizer,
};

/// An error that can occur when using the `WasmStanOptimizer`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error occurred while compiling the WebAssembly component.
    #[error("Error compiling component: {0}")]
    Compilation(wasmtime::Error),
    /// An error occurred while linking the WebAssembly component.
    #[error("Error linking component: {0}")]
    Linking(wasmtime::Error),
    /// An error occurred while instantiating the WebAssembly component.
    #[error("Error instantiating component: {0}")]
    Instantiation(wasmtime::Error),
    /// An error occurred in wasmtime while running the WebAssembly component.
    #[error("Error running component: {0}")]
    Runtime(wasmtime::Error),
    /// An error occurred in Stan while running the optimization.
    #[error("Error running optimization: {0}")]
    Optimize(String),
    /// An invalid parameter value was received from the optimizer.
    #[error("Invalid value ({value}) for parameter {param} received from optimizer")]
    InvalidParam {
        /// The parameter name.
        param: String,
        /// The value received from the optimizer.
        value: f64,
    },
}

#[allow(missing_docs)]
mod gen {
    use wasmtime::component::bindgen;

    bindgen!({
        world: "prophet-wasmstan",
        path: "prophet-wasmstan.wit",
        // Uncomment this to include the pregenerated file in the `target` directory
        // somewhere (search for `prophet-wasmstan0.rs`).
        // include_generated_code_from_file: true,
    });
}

use gen::*;

/// State required to run the WASI module. Will be stored in the `Store` while
/// the module is running.
struct WasiState {
    ctx: WasiCtx,
    table: ResourceTable,
}

impl Default for WasiState {
    fn default() -> Self {
        Self {
            ctx: WasiCtx::builder().build(),
            table: Default::default(),
        }
    }
}

/// View of the WASI state, required to call `wasmtime_wasi::add_to_linker_sync`.
impl WasiView for WasiState {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.ctx
    }
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

/// An `Optimizer` which runs the Prophet model inside a WebAssembly
/// component.
#[derive(Clone)]
pub struct WasmstanOptimizer {
    engine: Engine,
    instance_pre: ProphetWasmstanPre<WasiState>,
}

impl fmt::Debug for WasmstanOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmStanOptimizer").finish()
    }
}

impl Default for WasmstanOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmstanOptimizer {
    /// Create a new `WasmStanOptimizer`.
    ///
    /// This uses the WASM image embedded in the binary to create
    /// a WebAssembly component which is then linked to the WASI
    /// imports and pre-instantiated.
    ///
    /// # Panics
    ///
    /// This function panics if the embedded WASM image is invalid.
    /// This should never happen as the WASM is built and tested in CI,
    /// but if it does, please file a bug.
    #[cfg(feature = "wasmstan")]
    pub fn new() -> Self {
        Self::with_custom_image(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/prophet-wasmstan.wasm"
        )))
        .expect("embedded WASM image is invalid, this is a bug in augurs_prophet")
    }

    /// Create a new `WasmStanOptimizer` with a custom WebAssembly image.
    ///
    /// This allows you to use a custom WebAssembly image instead of the one
    /// embedded in the binary. This is useful if you want to use a different
    /// version of the model or if you don't want to embed the image in the
    /// binary.
    ///
    /// The custom image must be a WebAssembly component which satisfies the
    /// [WIT interface of the Prophet model][wit].
    ///
    /// Note that this does not accept the text format of WebAssembly.
    ///
    /// # Errors
    ///
    /// This function returns an error if the WebAssembly binary is invalid
    /// or if it does not satisfy the WIT interface of the Prophet model.
    ///
    /// [wit]: https://github.com/grafana/augurs/blob/main/components/cpp/prophet-wasmstan/wit/prophet-wasmstan.wit
    pub fn with_custom_image(binary: &[u8]) -> Result<Self, Error> {
        // Create an engine in which to compile and run everything.
        let engine = Engine::default();

        // Create a component from the compiled and embedded WASM binary.
        let component = Component::from_binary(&engine, binary).map_err(Error::Compilation)?;

        // Create a linker, which will add WASI imports to the component.
        let mut linker = Linker::new(&engine);
        wasmtime_wasi::add_to_linker_sync(&mut linker).map_err(Error::Linking)?;

        // Create a pre-instantiated component.
        // This does as much work as possible here, so that `optimize` can
        // be called multiple times with the minimum amount of overhead.
        let instance_pre = linker
            .instantiate_pre(&component)
            .and_then(ProphetWasmstanPre::new)
            .map_err(Error::Instantiation)?;
        Ok(Self {
            engine,
            instance_pre,
        })
    }

    /// Optimize the model using the given parameters.
    fn wasm_optimize(
        &self,
        init: &augurs::prophet_wasmstan::types::Inits,
        data: &String,
        opts: augurs::prophet_wasmstan::types::OptimizeOpts,
    ) -> Result<OptimizedParams, Error> {
        let mut store = Store::new(&self.engine, WasiState::default());
        let instance = self
            .instance_pre
            .instantiate(&mut store)
            .map_err(Error::Instantiation)?;
        instance
            .augurs_prophet_wasmstan_optimizer()
            .call_optimize(&mut store, init, data, opts)
            .map_err(Error::Runtime)?
            .map_err(Error::Optimize)
            .and_then(|op| op.params.try_into())
    }
}

impl TryFrom<augurs::prophet_wasmstan::types::OptimizedParams> for OptimizedParams {
    type Error = Error;
    fn try_from(
        value: augurs::prophet_wasmstan::types::OptimizedParams,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            k: value.k,
            m: value.m,
            sigma_obs: value
                .sigma_obs
                .try_into()
                .map_err(|_| Error::InvalidParam {
                    param: "sigma_obs".to_string(),
                    value: value.sigma_obs,
                })?,
            delta: value.delta,
            beta: value.beta,
            trend: value.trend,
        })
    }
}

impl Optimizer for WasmstanOptimizer {
    fn optimize(
        &self,
        init: &InitialParams,
        data: &Data,
        opts: &OptimizeOpts,
    ) -> Result<OptimizedParams, optimizer::Error> {
        let data = serde_json::to_string(&data).map_err(optimizer::Error::custom)?;
        self.wasm_optimize(&init.into(), &data, opts.into())
            .map_err(optimizer::Error::custom)
    }
}

impl From<&InitialParams> for augurs::prophet_wasmstan::types::Inits {
    fn from(init: &InitialParams) -> Self {
        Self {
            k: init.k,
            m: init.m,
            sigma_obs: init.sigma_obs.into(),
            delta: init.delta.clone(),
            beta: init.beta.clone(),
        }
    }
}

impl From<&OptimizeOpts> for augurs::prophet_wasmstan::types::OptimizeOpts {
    fn from(opts: &OptimizeOpts) -> Self {
        Self {
            algorithm: opts.algorithm.map(Into::into),
            seed: opts.seed,
            chain: opts.chain,
            init_alpha: opts.init_alpha,
            tol_obj: opts.tol_obj,
            tol_rel_obj: opts.tol_rel_obj,
            tol_grad: opts.tol_grad,
            tol_rel_grad: opts.tol_rel_grad,
            tol_param: opts.tol_param,
            history_size: opts.history_size,
            iter: opts.iter,
            jacobian: opts.jacobian,
            refresh: opts.refresh,
        }
    }
}

impl From<optimizer::Algorithm> for augurs::prophet_wasmstan::types::Algorithm {
    fn from(algo: optimizer::Algorithm) -> Self {
        match algo {
            optimizer::Algorithm::Bfgs => Self::Bfgs,
            optimizer::Algorithm::Lbfgs => Self::Lbfgs,
            optimizer::Algorithm::Newton => Self::Newton,
        }
    }
}
