// Package changepoint provides changepoint detection algorithms.
package changepoint

// Run `just go/build-wasm` first to produce the Wasm artifact referenced below.
//go:generate gravity --world changepoint --wit-file ../../components/rust/changepoint/wit/changepoint.wit --output changepoint.go ../../target/wasm32-unknown-unknown/release/augurs_changepoint_component.wasm
