set ignore-comments

# Build and publish the augurs-js package to npm with the @bsull scope.
publish-npm:
  cd crates/augurs-js && \
    wasm-pack build --release --scope bsull --out-name augurs --target web -- --features parallel && \
    node prepublish && \
    wasm-pack publish --access public

test:
  cargo nextest run --all-features --workspace

doctest:
  # Ignore augurs-js and pyaugurs since they either won't compile with all features enabled
  # or doesn't have any meaningful doctests anyway, since they're not published.
  cargo test --doc --all-features --workspace --exclude augurs-js --exclude pyaugurs
