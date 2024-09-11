#![doc = include_str!("../README.md")]
// Annoying, hopefully https://github.com/madonoharu/tsify/issues/42 will
// be resolved at some point.
#![allow(non_snake_case, clippy::empty_docs)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use serde::Serialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

/// Initialize the rayon thread pool.
///
/// This must be called once (from a Javascript context) and awaited
/// before using parallel mode of algorithms, to set up the thread pool.
///
/// # Example (JS)
///
/// ```js
/// // worker.ts
/// import init, { Dbscan, Dtw, initThreadPool} from '@bsull/augurs';
///
/// init().then(async () => {
///   console.debug('augurs initialized');
///   await initThreadPool(navigator.hardwareConcurrency * 2);
///   console.debug('augurs thread pool initialized');
/// });
///
/// export function dbscan(series: Float64Array[], epsilon: number, minClusterSize: number): number[] {
///   const distanceMatrix = Dtw.euclidean({ window: 10, parallelize: true }).distanceMatrix(series);
///   const clusterLabels = new Dbscan({ epsilon, minClusterSize }).fit(distanceMatrix);
///   return Array.from(clusterLabels);
/// }
///
/// // index.js
/// import { dbscan } from './worker';
///
/// async function runClustering(series: Float64Array[]): Promise<number[]> {
///   return dbscan(series, 0.1, 10);  // await only required if using workerize-loader
/// }
///
/// // or using e.g. workerize-loader to run in a dedicated worker:
/// import worker from 'workerize-loader?ready&name=augurs!./worker';
///
/// const instance = worker()
///
/// async function runClustering(series: Float64Array[]): Promise<number[]> {
///   await instance.ready;
///   return instance.dbscan(series, 0.1, 10);
/// }
/// ```
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

mod changepoints;
pub mod clustering;
mod dtw;
pub mod ets;
pub mod mstl;
mod outlier;
pub mod seasons;

/// Initialize the logger and panic hook.
///
/// This will be called automatically when the module is imported.
/// It sets the default tracing subscriber to `tracing-wasm`, and
/// sets WASM panics to print to the console with a helpful error
/// message.
#[wasm_bindgen(start)]
pub fn custom_init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    #[cfg(feature = "tracing-wasm")]
    tracing_wasm::try_set_as_global_default().ok();
}

// Wrapper types for the core types, so we can derive `Tsify` for them.
// This avoids having to worry about `tsify` in the `augurs-core` crate.

/// Forecast intervals.
#[derive(Clone, Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct ForecastIntervals {
    /// The confidence level for the intervals.
    pub level: f64,
    /// The lower prediction intervals.
    pub lower: Vec<f64>,
    /// The upper prediction intervals.
    pub upper: Vec<f64>,
}

impl From<augurs_core::ForecastIntervals> for ForecastIntervals {
    fn from(f: augurs_core::ForecastIntervals) -> Self {
        Self {
            level: f.level,
            lower: f.lower,
            upper: f.upper,
        }
    }
}

/// A forecast containing point forecasts and, optionally, prediction intervals.
#[derive(Clone, Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct Forecast {
    /// The point forecasts.
    pub point: Vec<f64>,
    /// The forecast intervals, if requested and supported
    /// by the trend model.
    pub intervals: Option<ForecastIntervals>,
}

impl From<augurs_core::Forecast> for Forecast {
    fn from(f: augurs_core::Forecast) -> Self {
        Self {
            point: f.point,
            intervals: f.intervals.map(Into::into),
        }
    }
}
