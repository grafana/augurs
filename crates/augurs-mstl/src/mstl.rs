//! The MSTL algorithm.
//!
//! This module contains the implementation of the MSTL algorithm.
//! The algorithm effectively runs multiple STL decompositions in
//! order of increasing seasonal period.
//!
//! This module is concerned with the decomposition itself. The
//! [`MSTLModel`][crate::MSTLModel] struct uses this module to perform
//! the decomposition before modeling the trend component and recombining
//! the components into a final forecast.

use std::collections::HashMap;

use stlrs::{StlParams, StlResult};
use tracing::instrument;

use crate::{Error, Result};

/// Multiple seasonal-trend decomposition of a time series.
///
/// This struct handles with the actual decomposition. Calling [`MSTL::fit`]
/// will run STL for each season length and return an object containing the
/// final trend, seasonal, and remainder components.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct MSTL<'a> {
    /// Time series to decompose.
    y: Vec<f32>,
    /// Periodicity of the seasonal components.
    periods: &'a mut Vec<usize>,
    /// Parameters for the STL decomposition.
    stl_params: StlParams,
}

impl<'a> MSTL<'a> {
    /// Create a new MSTL decomposition.
    ///
    /// Call `fit` to run the decomposition.
    pub fn new(y: impl Iterator<Item = f32>, periods: &'a mut Vec<usize>) -> Self {
        Self {
            y: y.collect::<Vec<_>>(),
            periods,
            stl_params: stlrs::params(),
        }
    }

    /// Set the parameters for each individual STL.
    pub fn stl_params(mut self, params: StlParams) -> Self {
        self.stl_params = params;
        self
    }

    /// Run the MSTL algorithm, returning the trend, seasonal, and remainder components.
    #[instrument(skip(self), level = "debug")]
    pub fn fit(mut self) -> Result<MSTLDecomposition> {
        self.process_periods()?;
        let seasonal_windows: Vec<usize> = self.seasonal_windows();
        let iterate = if self.periods.len() == 1 { 1 } else { 2 };

        let mut seasonals: HashMap<usize, StlResult> = HashMap::with_capacity(self.periods.len());
        // self.periods.iter().copied().map(|p| (p, None)).collect();
        let mut deseas = self.y;
        let mut res: Option<StlResult> = None;
        for i in 0..iterate {
            let zipped = self.periods.iter().zip(seasonal_windows.iter());
            for (period, seasonal_window) in zipped {
                let seas = seasonals.entry(*period);
                // Start by adding on the seasonal effect.
                if let std::collections::hash_map::Entry::Occupied(ref seas) = seas {
                    deseas
                        .iter_mut()
                        .zip(seas.get().seasonal().iter())
                        .for_each(|(d, s)| *d += *s);
                }
                // Decompose the time series for specific seasonal period.
                let fit =
                    tracing::debug_span!("STL.fit", i, seasonal_window, period).in_scope(|| {
                        self.stl_params
                            .seasonal_length(*seasonal_window)
                            .fit(&deseas, *period)
                    })?;
                // Subtract the seasonal effect again.
                deseas
                    .iter_mut()
                    .zip(fit.seasonal().iter())
                    .for_each(|(d, s)| *d -= *s);
                match seas {
                    std::collections::hash_map::Entry::Occupied(mut o) => {
                        o.insert(fit.clone());
                    }
                    std::collections::hash_map::Entry::Vacant(x) => {
                        x.insert(fit.clone());
                    }
                }
                res = Some(fit);
            }
        }
        let fit = res.ok_or_else(|| Error::MSTL("no STL fit".to_string()))?;
        let trend = fit.trend().to_vec();
        deseas
            .iter_mut()
            .zip(trend.iter())
            .for_each(|(d, r)| *d -= *r);
        let robust_weights = fit.weights().to_vec();
        Ok(MSTLDecomposition {
            trend,
            seasonal: seasonals
                .into_iter()
                .map(|(k, v)| (k, v.seasonal().to_vec()))
                .collect(),
            residuals: deseas,
            robust_weights,
        })
    }

    /// Return the default seasonal windows.
    ///
    /// This uses the formula from appendix A of the [MSTL paper].
    ///
    /// [MSTL paper]: https://arxiv.org/abs/2107.13462
    // TODO: make this configurable - the paper notes that "a smaller value of
    // s.window is set if the seasonal pattern evolves quickly, whereas a higher
    // value is used if the seasonal pattern is constant over time."
    fn seasonal_windows(&self) -> Vec<usize> {
        (0..self.periods.len()).map(|i| 7 + 4 * (i + 1)).collect()
    }

    /// Process the input periods.
    ///
    /// Specifically:
    /// 1. Sort periods in ascending order.
    /// 2. Ensure periods is non-empty and that all periods are > 1.
    /// 3. Remove periods greater than half of the time series.
    fn process_periods(&mut self) -> Result<()> {
        // Sort periods in ascending order to minimise seasonal confounding.
        self.periods.sort_unstable();
        // For now we don't support non-seasonal data.
        // TODO: write a supersmoother implementation to handle this case.
        if self.periods.is_empty() || self.periods.first().unwrap_or(&0) <= &1 {
            return Err(Error::MSTL("non-seasonal data not supported".to_string()));
        }
        // Check for and remove periods greater than half of the time series.
        self.periods.retain(|p| *p <= self.y.len() / 2);
        Ok(())
    }
}

/// The result of a MSTL decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(Default))]
pub struct MSTLDecomposition {
    /// Trend component.
    trend: Vec<f32>,
    /// Mapping from period to seasonal component.
    seasonal: HashMap<usize, Vec<f32>>,
    /// Residuals.
    residuals: Vec<f32>,
    /// Weights used in the robust fit.
    robust_weights: Vec<f32>,
}

impl MSTLDecomposition {
    /// Return the trend component.
    pub fn trend(&self) -> &[f32] {
        &self.trend
    }

    /// Return the seasonal component for a given period,
    /// or None if the period is not present.
    pub fn seasonal(&self, period: usize) -> Option<&[f32]> {
        self.seasonal.get(&period).map(|v| v.as_slice())
    }

    /// Return a mapping from period to seasonal component.
    pub fn seasonals(&self) -> &HashMap<usize, Vec<f32>> {
        &self.seasonal
    }

    /// Return the residuals.
    pub fn residuals(&self) -> &[f32] {
        &self.residuals
    }

    /// Return the robust weights.
    pub fn robust_weights(&self) -> &[f32] {
        &self.robust_weights
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use augurs_testing::data::VIC_ELEC;

    use super::*;

    fn vic_elec_results() -> MSTLDecomposition {
        let mut results = MSTLDecomposition::default();
        let data = include_str!("../data/vic_elec_results.csv");
        data.lines()
            .skip(1)
            .map(|l| l.split(',').map(|cell| cell.parse().unwrap()))
            .for_each(|mut row| {
                let _data = row.next().unwrap();
                results.trend.push(row.next().unwrap());
                results
                    .seasonal
                    .entry(24)
                    .or_default()
                    .push(row.next().unwrap());
                results
                    .seasonal
                    .entry(168)
                    .or_default()
                    .push(row.next().unwrap());
                results.residuals.push(row.next().unwrap());
            });
        results
    }

    #[test]
    fn results_match_r() {
        let y = &VIC_ELEC;
        let mut params = stlrs::params();
        params
            .seasonal_degree(0)
            .seasonal_jump(1)
            .trend_degree(1)
            .trend_jump(1)
            .low_pass_degree(1)
            .inner_loops(2)
            .outer_loops(0);
        let mut periods = vec![24, 24 * 7];
        let mstl = MSTL::new(y.iter().map(|&x| x as f32), &mut periods).stl_params(params);
        let res = mstl.fit().unwrap();

        let expected = vic_elec_results();
        res.trend()
            .iter()
            .zip(expected.trend().iter())
            .for_each(|(a, b)| assert_approx_eq!(a, b, 1.0));
        res.seasonal(24)
            .unwrap()
            .iter()
            .zip(expected.seasonal(24).unwrap().iter())
            // Some numeric instability somewhere causes this to differ by
            // up to 1.0 somewhere :/
            .for_each(|(&a, &b)| assert_approx_eq!(a, b, 1e1_f32));
        res.seasonal(168)
            .unwrap()
            .iter()
            .zip(expected.seasonal(168).unwrap().iter())
            .for_each(|(a, b)| assert_approx_eq!(a, b, 1e-1_f32));
        res.residuals()
            .iter()
            .zip(expected.residuals().iter())
            .for_each(|(a, b)| assert_approx_eq!(a, b, 1e1_f32));
    }
}
