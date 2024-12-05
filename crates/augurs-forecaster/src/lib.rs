#![doc = include_str!("../README.md")]

mod data;
mod error;
mod forecaster;
mod power_transforms;
pub mod transforms;

pub use data::Data;
pub use error::Error;
pub use forecaster::Forecaster;
pub use power_transforms::optimize_lambda;
pub use transforms::Transform;
pub use transforms::Transforms;

type Result<T> = std::result::Result<T, Error>;
