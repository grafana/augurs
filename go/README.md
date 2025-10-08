# Go bindings (based on WebAssembly)

This directory contains a Go package which is generated from the Wasm components in the `components/rust` directory of this repository. The Go code is generated using [`gravity`](https://github.com/arcjet/gravity).

As `gravity` improves, so will these bindings.

## Building

### Prerequisites

- A recent nightly Rust compiler (e.g. `rustup install nightly`)
- The `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)
- [Go](https://go.dev/)
- [gravity](https://github.com/arcjet/gravity)
- [just](https://just.systems/man/en/)

### Steps

1. Build the WebAssembly components:

   ```sh
   just build-wasm
   ```

2. Generate the Go packages

   ```sh
   just generate
   ```

3. Run the Go tests


   ```sh
   just test
   ```
