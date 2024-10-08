use rand::Rng;

use super::Error;

/// Implements the [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
/// distribution.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
    /// Constructs a new laplace distribution with the given
    /// location and scale.
    ///
    /// # Errors
    ///
    /// Returns an error if location or scale are `NaN` or `scale <= 0.0`
    pub(crate) fn new(location: f64, scale: f64) -> Result<Self, Error> {
        if location.is_nan() || scale.is_nan() || scale <= 0.0 {
            Err(Error)
        } else {
            Ok(Laplace { location, scale })
        }
    }
}

impl ::rand::distributions::Distribution<f64> for Laplace {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let mut x = rng.gen_range(-0.5_f64..0.5);
        if x == -0.5 {
            x += f64::EPSILON;
        }
        self.location - self.scale * x.signum() * (1. - 2. * x.abs()).ln()
    }
}
