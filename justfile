set ignore-comments

build-augurs-js:
  just js/build

test-augurs-js: build-augurs-js
  just js/test

# Build and publish the augurs JS package to npm with the @bsull scope.
publish-augurs-js: test-augurs-js
  just js/publish

test:
  cargo nextest run \
    --all-features \
    --workspace \
    --exclude augurs-js \
    --exclude pyaugurs

# Run all unit and integration tests, plus examples and benchmarks,
# except for those which require `iai` (which isn't available on
# all platforms) and the Prophet benchmarks which require a STAN
# installation.
test-all:
  cargo nextest run \
    --all-features \
    --all-targets \
    --workspace \
    --exclude augurs-changepoint-js \
    --exclude augurs-clustering-js \
    --exclude augurs-core-js \
    --exclude augurs-dtw-js \
    --exclude augurs-ets-js \
    --exclude augurs-mstl-js \
    --exclude augurs-prophet-js \
    --exclude augurs-seasons-js \
    --exclude pyaugurs \
    -E 'not (binary(/iai/) | binary(/prophet-cmdstan/))'

doctest:
  # Ignore augurs-js and pyaugurs since they either won't compile with all features enabled
  # or doesn't have any meaningful doctests anyway, since they're not published.
  cargo test --doc --all-features --workspace --exclude augurs-js --exclude pyaugurs

doc:
  cargo doc --all-features --workspace --exclude augurs-js --exclude pyaugurs --open

watch:
  bacon

# Download the Prophet Stan model.
download-prophet-stan-model:
  cargo run --features download --bin download-stan-model

build-component:
  just components/build
  cp components/cpp/prophet-wasmstan/prophet-wasmstan.wasm crates/augurs-prophet/prophet-wasmstan.wasm
