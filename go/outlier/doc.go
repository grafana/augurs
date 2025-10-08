// Package outlier provides outlier detection algorithms.
package outlier

// Run `just go/build-wasm` first to produce the Wasm artifact referenced below.
//go:generate gravity --world outlier --wit-file ../../components/rust/outlier/wit/outlier.wit --output outlier.go ../../target/wasm32-unknown-unknown/release/augurs_outlier_component.wasm
