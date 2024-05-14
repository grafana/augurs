use std::fmt;

pub use changepoint::rv::dist;
use changepoint::{
    rv::{
        dist::NormalGamma,
        traits::{ConjugatePrior, HasSuffStat, Rv},
    },
    utils::map_changepoints,
    BocpdLike, BocpdTruncated,
};
use itertools::Itertools;

pub type GaussianDetector = Detector<dist::Gaussian, dist::NormalGamma>;

#[derive(Debug, Clone)]
pub struct Detector<Dist, Prior>
where
    Dist: Rv<f64> + HasSuffStat<f64>,
    Prior: ConjugatePrior<f64, Dist> + Clone,
    Dist::Stat: Clone + fmt::Debug,
{
    detector: BocpdTruncated<f64, Dist, Prior>,
}

impl<Dist, Prior> Detector<Dist, Prior>
where
    Dist: Rv<f64> + HasSuffStat<f64>,
    Prior: ConjugatePrior<f64, Dist, Posterior = Prior> + Clone,
    Dist::Stat: Clone + fmt::Debug,
{
    pub fn detect_changepoints(&mut self, y: &[f64]) -> Vec<usize> {
        let run_lengths = y
            .iter()
            .map(|d| self.detector.step(d).to_vec())
            .collect_vec();
        map_changepoints(&run_lengths)
    }
}

impl Detector<dist::Gaussian, dist::NormalGamma> {
    pub fn gaussian(hazard: f64, prior: NormalGamma) -> Self {
        Self {
            detector: BocpdTruncated::new(hazard, prior),
        }
    }
}

impl Default for Detector<dist::Gaussian, dist::NormalGamma> {
    fn default() -> Self {
        Self::gaussian(
            // TODO: figure out a good value for this.
            250.0,
            // TODO: figure out a good value for this.
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
        )
    }
}
