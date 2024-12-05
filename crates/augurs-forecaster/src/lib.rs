#![doc = include_str!("../README.md")]

mod data;
mod error;
mod forecaster;
mod power_transforms;
pub mod transforms;

pub use data::Data;
pub use error::Error;
pub use forecaster::Forecaster;
pub use transforms::Transform;
pub use transforms::Transforms;
pub use power_transforms::optimize_lambda;


type Result<T> = std::result::Result<T, Error>;
