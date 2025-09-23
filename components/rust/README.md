# Rust Wasm Components

This directory contains various Wasm components implemented in Rust using the augurs library.

The resulting Wasm files can be used either directly or by generating host bindings for a specific language using e.g. [`gravity`](https://github.com/arcjet/gravity).

See the `go` directory for a Go package which is generated from these components. That directory also contains a `justfile` for building the components, which requires some specific `RUSTFLAGS`.
