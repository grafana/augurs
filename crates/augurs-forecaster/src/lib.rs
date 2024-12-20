#![doc = include_str!("../README.md")]

mod data;
mod error;
mod forecaster;
pub mod transforms;

pub use data::Data;
pub use error::Error;
pub use forecaster::Forecaster;
pub use transforms::Transform;
pub(crate) use transforms::Transforms;

type Result<T> = std::result::Result<T, Error>;
