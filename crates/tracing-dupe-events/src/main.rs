fn main() {
    tracing_subscriber::fmt::init();
    tracing::info!("Hello, world, hopefully only once");
}
