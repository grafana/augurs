//! Distributions required for the Prophet model.
//!
//! This is copied straight out of the [`statrs`] crate; full credit goes to
//! the original authors.
//!
//! [`statrs`]: https://crates.io/crates/statrs

mod laplace;
mod normal;
mod poisson;

pub(crate) use laplace::Laplace;
pub(crate) use normal::Normal;
pub(crate) use poisson::Poisson;

#[derive(Debug, Clone, thiserror::Error)]
#[error("Distribution error: bad parameters")]
pub(crate) struct Error;

mod consts {
    #![allow(clippy::excessive_precision)]
    /// Auxiliary variable when evaluating the `gamma_ln` function
    pub(crate) const GAMMA_R: f64 = 10.900511;

    /// Polynomial coefficients for approximating the `gamma_ln` function
    pub(crate) const GAMMA_DK: &[f64] = &[
        2.48574089138753565546e-5,
        1.05142378581721974210,
        -3.45687097222016235469,
        4.51227709466894823700,
        -2.98285225323576655721,
        1.05639711577126713077,
        -1.95428773191645869583e-1,
        1.70970543404441224307e-2,
        -5.71926117404305781283e-4,
        4.63399473359905636708e-6,
        -2.71994908488607703910e-9,
    ];

    /// The maximum factorial representable
    /// by a 64-bit floating point without
    /// overflowing
    pub(crate) const MAX_FACTORIAL: usize = 170;

    /// Constant value for `ln(pi)`
    pub(crate) const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;

    /// Constant value for `ln(2 * sqrt(e / pi))`
    pub(crate) const LN_2_SQRT_E_OVER_PI: f64 =
        0.6207822376352452223455184457816472122518527279025978;
}
