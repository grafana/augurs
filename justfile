# Set up npm to use the Grafana Labs npm registry for the @grafana-ml scope.
authenticate-npm-dev:
  npm set @grafana-ml:registry=https://us-npm.pkg.dev/grafanalabs-dev/ml-npm-dev/
  npx google-artifactregistry-auth

# Build and publish the augurs-js package to the Grafana Labs npm registry with the @grafana-ml scope.
publish-npm-dev:
  cd crates/augurs-js && \
    wasm-pack build --release --scope grafana-ml && \
    wasm-pack publish --access public

# Build and publish the augurs-js package to npm with the @bsull scope.
publish-npm:
  cd crates/augurs-js && \
    wasm-pack build --release --scope bsull && \
    wasm-pack publish --access public

