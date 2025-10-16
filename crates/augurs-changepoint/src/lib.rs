#![doc = include_str!("../README.md")]
use std::{fmt, num::NonZeroUsize};

pub use changepoint::rv::{dist, process::gaussian::kernel};
use changepoint::{
    rv::{
        dist::{Gaussian, NormalGamma, NormalInvGamma},
        process::gaussian::kernel::{
            AddKernel, ConstantKernel, Kernel, ProductKernel, RBFKernel, WhiteKernel,
        },
        traits::{ConjugatePrior, HasDensity, HasSuffStat, Rv},
    },
    utils::map_changepoints,
    Argpcp, BocpdLike, BocpdTruncated,
};
use itertools::Itertools;

/// Trait implemented by changepoint detectors.
pub trait Detector {
    /// Detect changepoints in the provided vector, returning their indices.
    fn detect_changepoints(&mut self, y: &[f64]) -> Vec<usize>;
}

/// A changepoint detector using Bayesian Online Changepoint Detection.
///
/// Based on [this paper][paper], using the implementation from the
/// [changepoint] crate.
///
/// [changepoint]: https://crates.io/crates/changepoint
/// [paper]: https://arxiv.org/abs/0710.3742
#[derive(Debug, Clone)]
pub struct BocpdDetector<Dist, Prior>
where
    Dist: Rv<f64> + HasSuffStat<f64>,
    Prior: ConjugatePrior<f64, Dist> + HasDensity<Dist> + Clone,
    Dist::Stat: Clone + fmt::Debug,
{
    detector: BocpdTruncated<f64, Dist, Prior>,
}

impl<Dist, Prior> Detector for BocpdDetector<Dist, Prior>
where
    Dist: Rv<f64> + HasSuffStat<f64>,
    Prior: ConjugatePrior<f64, Dist, Posterior = Prior> + HasDensity<Dist> + Clone,
    Dist::Stat: Clone + fmt::Debug,
{
    fn detect_changepoints(&mut self, y: &[f64]) -> Vec<usize> {
        let run_lengths = y
            .iter()
            .map(|d| self.detector.step(d).to_vec())
            .collect_vec();
        map_changepoints(&run_lengths)
    }
}

/// A [`BocpdDetector`] for Normal data with a Normal Gamma prior.
pub type NormalGammaDetector = BocpdDetector<Gaussian, NormalGamma>;
/// A [`BocpdDetector`] for Normal data with a Normal inverse-Gamma prior.
pub type NormalInvGammaDetector = BocpdDetector<Gaussian, NormalInvGamma>;

impl NormalGammaDetector {
    /// Create a detector for Normal data using the given hazard_lambda and prior.
    pub fn normal_gamma(hazard_lambda: f64, prior: NormalGamma) -> Self {
        Self {
            detector: BocpdTruncated::new(hazard_lambda, prior),
        }
    }
}

impl NormalInvGammaDetector {
    /// Create a detector for Normal data using the given hazard_lambda and prior.
    pub fn normal_inv_gamma(hazard_lambda: f64, prior: NormalInvGamma) -> Self {
        Self {
            detector: BocpdTruncated::new(hazard_lambda, prior),
        }
    }
}

impl Default for NormalGammaDetector {
    fn default() -> Self {
        Self::normal_gamma(250.0, NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0))
    }
}

impl Default for NormalInvGammaDetector {
    fn default() -> Self {
        Self::normal_inv_gamma(250.0, NormalInvGamma::new_unchecked(0.0, 1.0, 1.0, 1.0))
    }
}

type DefaultKernel = AddKernel<ProductKernel<ConstantKernel, RBFKernel>, WhiteKernel>;
/// An [`ArgpcpDetector`] with a sensible default choice of kernel.
pub type DefaultArgpcpDetector = ArgpcpDetector<DefaultKernel>;

/// A changepoint detector using autoregressive Gaussian Processes.
///
/// Based on [Ryan Turner's thesis][thesis], using the implementation from
/// the [changepoint] crate.
///
/// [thesis]: https://www.repository.cam.ac.uk/bitstream/handle/1810/242181/thesis.pdf?sequence=1&isAllowed=y
/// [changepoint]: https://crates.io/crates/changepoint
#[derive(Debug, Clone)]
pub struct ArgpcpDetector<K>
where
    K: Kernel,
{
    detector: Argpcp<K>,
}

impl<K> From<Argpcp<K>> for ArgpcpDetector<K>
where
    K: Kernel,
{
    fn from(detector: Argpcp<K>) -> Self {
        Self { detector }
    }
}

impl DefaultArgpcpDetector {
    /// Get a builder to configure the parameters of the detector.
    pub fn builder() -> DefaultArgpcpDetectorBuilder {
        DefaultArgpcpDetectorBuilder::default()
    }
}

impl<K> Detector for ArgpcpDetector<K>
where
    K: Kernel,
{
    fn detect_changepoints(&mut self, y: &[f64]) -> Vec<usize> {
        let run_lengths = y
            .iter()
            .map(|d| self.detector.step(d).to_vec())
            .collect_vec();
        map_changepoints(&run_lengths)
    }
}

impl Default for DefaultArgpcpDetector {
    fn default() -> Self {
        DefaultArgpcpDetectorBuilder::default().build()
    }
}

/// Builder for a [`DefaultArgpcpDetector`].
#[derive(Debug, Clone)]
pub struct DefaultArgpcpDetectorBuilder {
    constant_value: f64,
    length_scale: f64,
    noise_level: f64,
    max_lag: NonZeroUsize,
    alpha0: f64,
    beta0: f64,
    logistic_hazard_h: f64,
    logistic_hazard_a: f64,
    logistic_hazard_b: f64,
}

impl DefaultArgpcpDetectorBuilder {
    /// Set the value for the constant kernel.
    pub fn constant_value(mut self, cv: f64) -> Self {
        self.constant_value = cv;
        self
    }
}

impl Default for DefaultArgpcpDetectorBuilder {
    fn default() -> Self {
        Self {
            constant_value: 0.5,
            length_scale: 10.0,
            noise_level: 0.01,
            max_lag: NonZeroUsize::new(3).unwrap(),
            alpha0: 2.0,
            beta0: 1.0,
            logistic_hazard_h: -5.0,
            logistic_hazard_a: 1.0,
            logistic_hazard_b: 1.0,
        }
    }
}

impl DefaultArgpcpDetectorBuilder {
    /// Build this [`DefaultArgpcpDetector`].
    pub fn build(self) -> DefaultArgpcpDetector {
        DefaultArgpcpDetector {
            detector: Argpcp::new(
                ConstantKernel::new_unchecked(self.constant_value)
                    * RBFKernel::new_unchecked(self.length_scale)
                    + WhiteKernel::new_unchecked(self.noise_level),
                self.max_lag.into(),
                self.alpha0,
                self.beta0,
                self.logistic_hazard_h,
                self.logistic_hazard_a,
                self.logistic_hazard_b,
            ),
        }
    }
}
