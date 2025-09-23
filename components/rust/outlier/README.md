# Outlier Detection Wasm Component

This directory contains:

- the interface definition for a Wasm component that can perform outlier detection on time series (see `wit/world.wit`)
- a Rust crate which exposes augurs' outlier detection functionality as a Wasm component (see `src/lib.rs`)
- tests for a Go package which can be generated from the Wasm component using [gravity] (see `outlier_test.go`)

The goal is to provide a Go package that can embed some augurs functionality without having to resort to FFI.

## Building

### Prerequisites

- A recent nightly Rust compiler (e.g. `rustup install nightly`)
- The `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)
- [Go](https://go.dev/)
- [gravity](https://github.com/arcjet/gravity)
- [just](https://just.systems/man/en/)

### Steps

1. Build the Rust crate

   ```sh
   just build-wasm
   ```

2. Generate the Go package

   ```sh
   just gen-go
   ```

3. Run the Go tests


   ```sh
   just test-go
   ```
