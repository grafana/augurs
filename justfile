set ignore-comments

build-augurs-js:
  rm -rf js/augurs/*
  just js/build

test-augurs-js: build-augurs-js
  just js/test

# Build and publish the augurs JS package to npm with the @bsull scope.
publish-augurs-js: test-augurs-js
  just js/publish

# Run unit tests
test:
  cargo nextest run \
    --all-features \
    --workspace \
    --exclude *-js \
    --exclude pyaugurs

# Run all unit and integration tests, plus examples, except for those which require `iai` (which isn't available on all platforms) and the Prophet benchmarks which require a STAN installation.
test-all:
  cargo nextest run \
    --all-features \
    --all-targets \
    --workspace \
    --exclude *-js \
    --exclude pyaugurs \
    -E 'not (binary(/iai/) | binary(/prophet-cmdstan/) | kind(bench))'

# Run benchmarks in "test mode" (but with optimizations) using nextest.
test-bench:
  cargo nextest run \
    --release \
    --all-features \
    --benches \
    --workspace \
    --exclude *-js \
    --exclude pyaugurs \
    -E 'not (binary(/iai/) | binary(/prophet-cmdstan/))'

doctest:
  # Ignore JS and pyaugurs crates since they either won't compile with all features enabled
  # or doesn't have any meaningful doctests anyway, since they're not published.
  cargo test \
    --doc \
    --all-features \
    --workspace \
    --exclude *-js \
    --exclude pyaugurs \

doc:
  cargo doc --all-features --workspace --exclude *-js --exclude pyaugurs --open

# Build the Python package
build-pyaugurs:
  cd crates/pyaugurs && uv pip install -e . --force-reinstall --no-deps

# Run Python tests
test-pyaugurs: build-pyaugurs
  cd crates/pyaugurs && uv run pytest tests/ -v

watch:
  bacon

# Download the Prophet Stan model.
download-prophet-stan-model:
  cargo run --features download --bin download-stan-model

copy-component-wasm:
  cp components/cpp/prophet-wasmstan/prophet-wasmstan.wasm crates/augurs-prophet/prophet-wasmstan.wasm

# Rebuild the prophet-wasmstan WASM component. Requires a local runner with the `act` tool.
rebuild-component:
  act --bind --artifact-server-path=/tmp/artifacts -W ./.github/workflows/wasmstan.yml
