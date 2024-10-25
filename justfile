set ignore-comments

build-augurs-js:
  cd crates/augurs-js && \
    rm -rf ./pkg && \
    wasm-pack build --scope bsull --out-name augurs --release --target web -- --features parallel

test-augurs-js: build-augurs-js
  cd crates/augurs-js/testpkg && \
    npm ci && \
    npm run typecheck && \
    npm run test:ci

# Build and publish the augurs JS package to npm with the @bsull scope.
publish-augurs-js: test-augurs-js
  cd crates/augurs-js && \
    node prepublish && \
    wasm-pack publish --access public

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
    --exclude augurs-js \
    --exclude pyaugurs \
    -E 'not (binary(/iai/) | binary(/prophet/))'

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
