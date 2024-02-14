#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use wasm_bindgen::prelude::*;

pub mod ets;
pub mod mstl;
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
