# Contributing

Thank you for your interest in contributing to augurs! We welcome contributions from everyone.

The augurs repository can be found [on GitHub][repo].

## Ways to Contribute

There are many ways to contribute:
- Report bugs and request features in the issue tracker
- Submit pull requests to fix issues or add features
- Improve documentation
- Share your experiences using augurs

## Development Setup

1. Fork and clone the repository
2. Install Rust via rustup if you haven't already
3. Install [`just`][just] for running tasks
4. Install [`cargo-binstall`][binstall] to install dependencies
5. Install dependencies:
```bash
(cd components && just install-deps)
```
6. Build the WASM component:
```bash
just build-component
```
7. Start building and checking the project using [bacon]:
```bash
just watch
```
8. Run tests using [nextest]:
```bash
just test
```

## Pull Request Process

1. Fork the repository and create a new branch from `main`
2. Make your changes and add tests if applicable
3. Update documentation as needed
4. Run tests and clippy to ensure everything passes
5. Submit a pull request describing your changes

## Code Style

- Follow Rust standard style guidelines
- Run `cargo fmt` before committing
- Run `cargo clippy` and address any warnings
- Add documentation comments for public APIs

## Testing

- Add tests for new functionality
- Ensure existing tests pass
- Include both unit tests and integration tests where appropriate

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT license OR APACHE 2.0 license).

[repo]: https://github.com/grafana/augurs/
[just]: https://just.systems/man/en/
[binstall]: https://github.com/cargo-bins/cargo-binstall
[bacon]: https://dystroy.org/bacon/
[nextest]: https://nexte.st/
