use crate::model::{
    ErrorComponent, ModelType, OptimizationCriteria, OptimizeParams, Params, SeasonalComponent,
    TrendComponent,
};

const TOLERANCE: f64 = 1e-10;
const HUGEN: f64 = 1e10;

#[derive(Debug, Clone)]
pub struct FitState {
    x: Vec<f64>,
    params: Params,
    lik: f64,
    amse: Vec<f64>,

    n_states: usize,

    // The length of the vector being passed to the Nelder-Mead optimizer? So:
    // - 1 for level (always)
    // - 1 for growth (if trend is not none)
    // - 1 for each seasonal parameter (if seasonal is not none)
    // - 1 for each of alpha/beta/gamma/phi being optimized.
    n_params: usize,

    residuals: Vec<f64>,
    fitted: Vec<f64>,
}

impl FitState {
    /// The likelihood of the model, given the data.
    #[inline]
    pub fn likelihood(&self) -> f64 {
        self.lik
    }

    /// The mean squared error (MSE) of the model.
    #[inline]
    pub fn mse(&self) -> f64 {
        self.amse[0]
    }

    /// The average mean squared error (AMSE) of the model.
    ///
    /// This is the average of the MSE over the number of forecasting horizons (`nmse`).
    #[inline]
    pub fn amse(&self) -> f64 {
        self.amse.iter().sum::<f64>() / self.amse.len() as f64
    }

    #[inline]
    pub fn sigma_squared(&self) -> f64 {
        self.residuals.iter().map(|e| e.powi(2)).sum::<f64>()
            / (self.residuals.len() as f64 - self.n_params as f64 - 2.0)
    }

    /// The mean absolute error (MAE) of the model.
    #[inline]
    pub fn mae(&self) -> f64 {
        self.residuals.iter().map(|e| e.abs()).sum::<f64>() / self.residuals.len() as f64
    }

    /// Returns an iterator over the states of the model.
    ///
    /// Each element is a slice of length `n_states` containing the level,
    /// growth and seasonal parameters of the model.
    pub fn states(&self) -> impl Iterator<Item = &[f64]> {
        self.x.chunks_exact(self.n_states)
    }

    /// The last state of the model.
    ///
    /// This should be the best value found while fitting the model,
    /// after which training stopped.
    pub fn last_state(&self) -> &[f64] {
        self.states().last().unwrap()
    }

    /// The parameters used when fitting the model.
    #[inline]
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The number of parameters used when fitting the model.
    #[inline]
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// The residuals of the model against the training data.
    #[inline]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// The fitted values of the model.
    #[inline]
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }
}

/// A full specification of an ETS model.
///
/// This includes everything specified by the user, but not the state of the
/// model (which is stored in [`FitState`]).
#[derive(Debug, Clone)]
pub struct Ets {
    pub model_type: ModelType,
    pub season_length: usize,
    pub damped: bool,
    pub nmse: usize,
    pub params: Params,
    pub opt_crit: OptimizationCriteria,
    // A few internal parameters used for fitting/forecasting the model.
    pub(crate) n_states: usize,
    pub(crate) opt_params: OptimizeParams,
}

impl Ets {
    /// Create a new ETS model.
    pub(crate) fn new(
        model_type: ModelType,
        damped: bool,
        nmse: usize,
        n_states: usize,
        params: Params,
        opt_params: OptimizeParams,
        opt_crit: OptimizationCriteria,
    ) -> Ets {
        Self {
            season_length: model_type.season.season_length(),
            model_type,
            damped,
            nmse,
            n_states,
            params,
            opt_params,
            opt_crit,
        }
    }

    /// Run the ETS calculation for the given data, state and parameters.
    ///
    /// This is the internal implementation of `etscalc`, which takes a mutable
    /// reference to the state vector and any output vectors. Many of these
    /// outputs are not used when fitting the model because all we need is
    /// the model's likelihood, so we can avoid reallocating them on each
    /// iteration of the optimization algorithm.
    ///
    /// Note that the return value of this function does not have the `x`,
    /// `residuals`, `fitted` or `amse` fields set. These must be set by the
    /// caller using the values passed in as arguments, if they are needed.
    // Allow clippy to complain about the number of arguments, because this
    // function is only used internally and it's easier to pass everything
    // in as arguments than to try to make it more readable by passing in
    // a struct.
    //
    // Steps:
    //
    // 1. Extract values of l, b, s from x.
    // 2. Iterate over the data.
    // 3. Forecast the next `nmse` values using current l, b, s.
    // 4. Update the `i`th residual using the forecast and the `i`th data point.
    // 5. Calculate the AMSE using `y` and the forecasts.
    // 6. Calculate the likelihood using the residuals.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn etscalc_in(
        &self,
        y: &[f64],
        x: &mut [f64],
        params: Params,
        residuals: &mut [f64],
        forecasts: &mut [f64],
        amse: &mut [f64],
        denom: &mut [f64],
        update_amse: bool,
    ) -> FitState {
        let Ets {
            model_type: ModelType { error, trend, .. },
            nmse,
            opt_params,
            n_states,
            ..
        } = self;

        // SAFETY: models will always include an error term, so `x` will have at least one element.
        let mut l = unsafe { *x.get_unchecked(0) };
        let mut b = trend.included().then_some(unsafe { *x.get_unchecked(1) });
        // let s = season.included().then_some(&input[2..self.n_states]);

        let mut lik = 0.0;
        let mut lik2 = 0.0;

        let n = y.len();
        for (i, (y_i, e_i)) in y.iter().copied().zip(residuals).enumerate() {
            let old_l = l;
            let old_b = b;

            let f_0 = self.forecast(params.phi, old_l, old_b, forecasts, *nmse);
            *e_i = self.compute_error(y_i, f_0);

            if update_amse {
                self.update_amse(y, i, forecasts, amse, denom);
            }

            (l, b) = self.updated_state(&params, old_l, old_b, y_i);

            // Update state vector.
            unsafe { *x.get_unchecked_mut(n_states * (i + 1)) = l };
            if let Some(b) = b {
                unsafe { *x.get_unchecked_mut(n_states * (i + 1) + 1) = b };
            }

            // Update likelihood.
            lik += e_i.powi(2);
            let val = f_0.abs();
            if val > 0.0 {
                lik2 += val.ln();
            } else {
                lik2 += (val + 1e-8).ln();
            }
        }

        if lik > 0.0 {
            lik = n as f64 * lik.ln();
        } else {
            lik = n as f64 * (lik + 1e-8).ln();
        }

        if error == &ErrorComponent::Multiplicative {
            lik += 2.0 * lik2
        }

        FitState {
            n_states: *n_states,
            params,
            lik,
            // We only care about x, amse, fitted values and residuals for the final state.
            // Leave them empty here; they're populated in `etscalc`.
            x: vec![],
            amse: vec![],
            fitted: vec![],
            residuals: vec![],
            n_params: n_states + opt_params.n_included(),
        }
    }

    /// Fit the ETS model to the given data, using the given initial state and
    /// parameters.
    // Allow clippy to complain about the number of arguments, because this
    // function is only used internally and it's easier to pass everything
    // in as arguments than to try to make it more readable by passing in
    // a struct.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pegels_resid_in(
        &self,
        y: &[f64],
        inputs: &[f64],
        mut x: Vec<f64>,
        mut params: Params,
        mut residuals: Vec<f64>,
        mut forecasts: Vec<f64>,
        mut amse: Vec<f64>,
        mut denom: Vec<f64>,
    ) -> FitState {
        let Self {
            opt_params,
            n_states,
            model_type,
            ..
        } = self;

        // If we've optimized params, they'll be included the inputs to the
        // optimizer, so use them to override the defaults.
        let mut i = 0;
        if opt_params.alpha {
            params.alpha = inputs[i];
            i += 1;
        }
        if opt_params.beta {
            params.beta = inputs[i];
            i += 1;
        }
        if opt_params.gamma {
            params.gamma = inputs[i];
            i += 1;
        }
        if opt_params.phi {
            params.phi = inputs[i];
            i += 1;
        }

        // The remaining parameters are the initial state.
        let state_inputs = &inputs[i..];
        x.truncate(state_inputs.len());
        x.copy_from_slice(state_inputs);
        x.resize(n_states * (y.len() + 1), 0.0);

        // Make final adjustments of params.
        if !self.damped {
            params.phi = 1.0;
        }
        if model_type.trend == TrendComponent::None {
            params.beta = 0.0;
        }
        if model_type.season == SeasonalComponent::None {
            params.gamma = 0.0;
        }

        let mut fit = self.etscalc_in(
            y,
            &mut x,
            params,
            &mut residuals,
            &mut forecasts,
            &mut amse,
            &mut denom,
            true,
        );
        if !fit.lik.is_nan() && (fit.lik + 99999.0).abs() < 1e-7 {
            fit.lik = f64::NAN;
        }
        fit.x = x;
        fit.residuals = residuals;
        fit.amse = amse;
        fit.fitted = match self.model_type.error {
            ErrorComponent::Additive => y
                .iter()
                .zip(fit.residuals().iter())
                .map(|(y, r)| y - r)
                .collect(),
            ErrorComponent::Multiplicative => y
                .iter()
                .zip(fit.residuals().iter())
                .map(|(y, r)| y / (1.0 + r))
                .collect(),
        };
        fit
    }

    /// Simulate values from the ETS model.
    ///
    /// This is currently untested as the codepath hasn't been hit yet...
    ///
    /// # Panics
    ///
    /// Panics if `yhat` is not the same length as `e` or if `f` is empty.
    pub(crate) fn etssimulate(
        &self,
        x: &[f64],
        params: Params,
        e: &[f64],
        f: &mut [f64],
        yhat: &mut [f64],
    ) {
        debug_assert!(yhat.len() == e.len());
        debug_assert!(!f.is_empty());
        let ModelType { error, trend, .. } = &self.model_type;

        let mut l = x[0];
        let mut b = trend.included().then_some(x[1]);

        for (y, e) in yhat.iter_mut().zip(e) {
            let old_l = l;
            let old_b = b;
            // One step forecast.
            self.forecast(params.phi, old_l, old_b, f, 1);
            // Set y using forecast.
            if f[0].abs() < TOLERANCE {
                *y = f64::NAN;
                return;
            }
            if error == &ErrorComponent::Additive {
                *y = f[0] + e;
            } else {
                *y = f[0] * (1.0 + e);
            }
            // Update state.
            (l, b) = self.updated_state(&params, old_l, old_b, *y);
        }
    }

    /// Calculate the forecasts for the given values of `l` and `b`.
    ///
    /// # Panics
    ///
    /// This function panics if `f.len() < horizon`.
    #[inline]
    pub(crate) fn forecast(
        &self,
        phi: f64,
        l: f64,
        b: Option<f64>,
        f: &mut [f64],
        horizon: usize,
    ) -> f64 {
        debug_assert!(f.len() >= horizon);
        let mut phi_star = phi;
        for (i, f_i) in f.iter_mut().take(horizon).enumerate() {
            if self.model_type.trend == TrendComponent::None {
                *f_i = l;
            } else if self.model_type.trend == TrendComponent::Additive {
                if let Some(b) = b {
                    *f_i = l + phi_star * b;
                } else {
                    *f_i = f64::NAN;
                }
            } else if let Some(b) = b {
                *f_i = l * b.powf(phi_star);
            }

            // let mut j: isize = self.season_length as isize - 1 - i as isize;
            // while j < 0 {
            //     j += self.season_length as isize;
            // }
            // TODO: seasonal component.
            if i < horizon - 1 {
                if (phi - 1.0).abs() < 1e-10 {
                    phi_star += 1.0;
                } else {
                    phi_star += phi.powi(i as i32 + 1);
                }
            }
        }
        f[0]
    }

    /// Compute the `i`th error term by comparing the `i`th observation to the first forecast
    /// at this point.
    #[inline]
    fn compute_error(&self, y_i: f64, f_0: f64) -> f64 {
        match self.model_type.error {
            ErrorComponent::Additive => y_i - f_0,
            ErrorComponent::Multiplicative => {
                let f_0_mod = if f_0.abs() < 1e-10 { f_0 + 1e-10 } else { f_0 };
                (y_i - f_0) / f_0_mod
            }
        }
    }

    /// Update the AMSE and denominator for the given forecasts.
    #[inline]
    fn update_amse(
        &self,
        y: &[f64],
        i: usize,
        forecasts: &[f64],
        amse: &mut [f64],
        denom: &mut [f64],
    ) {
        let n = y.len();
        for (j, ((a, d), f)) in amse[..self.nmse]
            .iter_mut()
            .zip(&mut denom[..self.nmse])
            .zip(&forecasts[..self.nmse])
            .enumerate()
        {
            if i + j < n {
                *d += 1.0;
                // Safety: we know that `i + j` is a valid index into `y`, as we
                // checked that `i + j` < n above.
                let tmp = unsafe { y.get_unchecked(i + j) } - f;
                *a = (*a * (*d - 1.0) + tmp.powi(2)) / *d;
            }
        }
    }

    /// Return the updated state (level and growth) for the given parameters.
    #[inline]
    fn updated_state(
        &self,
        params: &Params,
        old_l: f64,
        old_b: Option<f64>,
        y: f64,
    ) -> (f64, Option<f64>) {
        let (q, phi_b, p) = self.updated_level(params.phi, old_l, old_b, y);
        let l = q + params.alpha * (p - q);
        let b = self.updated_growth(params.alpha, params.beta, old_l, l, phi_b);
        (l, b)
    }

    /// Calculate the new level component.
    ///
    /// Returns the values of `q` and `phi_b`.
    ///
    /// TODO: update this to handle seasonality, too.
    #[inline]
    fn updated_level(&self, phi: f64, old_l: f64, old_b: Option<f64>, y: f64) -> (f64, f64, f64) {
        let trend = self.model_type.trend;
        let (q, phi_b) = match old_b {
            None => (old_l, 0.0),
            Some(old_b) if trend == TrendComponent::Additive => {
                let phi_b = phi * old_b;
                (old_l + phi_b, phi_b)
            }
            Some(old_b)
                if trend == TrendComponent::Multiplicative && (phi - 1.0).abs() < TOLERANCE =>
            {
                (old_l * old_b, old_b)
            }
            Some(old_b) if trend == TrendComponent::Multiplicative => {
                let phi_b = old_b.powf(phi);
                (old_l * phi_b, phi_b)
            }
            _ => unreachable!(),
        };
        // TODO: seasonal component.
        let p = y;
        (q, phi_b, p)
    }

    /// Calculate the new growth component.
    #[inline]
    fn updated_growth(&self, alpha: f64, beta: f64, old_l: f64, l: f64, phi_b: f64) -> Option<f64> {
        let r = match self.model_type.trend {
            TrendComponent::None => return None,
            TrendComponent::Additive => l - old_l,
            TrendComponent::Multiplicative if old_l.abs() < TOLERANCE => HUGEN,
            TrendComponent::Multiplicative => l / old_l,
        };
        Some(phi_b + (beta / alpha) * (r - phi_b))
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        data::AIR_PASSENGERS as AP,
        model::{
            ErrorComponent, ModelType, OptimizationCriteria, OptimizeParams, Params,
            SeasonalComponent, TrendComponent,
        },
    };

    use super::Ets;

    #[test]
    fn air_passengers_etscalc() {
        let mut init_states = vec![0.0; AP.len() * (2 + 1)];
        init_states[0] = 118.466667;
        init_states[1] = 2.060606;
        let params = Params {
            alpha: 0.016763333,
            beta: 0.017663333,
            gamma: 0.0,
            phi: 0.0,
        };
        let ets = Ets {
            season_length: 12,
            n_states: 2,
            damped: false,
            model_type: ModelType {
                error: ErrorComponent::Additive,
                trend: TrendComponent::Additive,
                season: SeasonalComponent::None,
            },
            nmse: 3,
            params: params.clone(),
            opt_params: OptimizeParams::default(),
            opt_crit: OptimizationCriteria::Likelihood,
        };
        let mut residuals = vec![0.0; AP.len()];
        let mut forecasts = vec![0.0; 3];
        let mut amse = vec![0.0; 3];
        let mut denom = vec![0.0; 3];
        let fit = ets.etscalc_in(
            &AP,
            &mut init_states,
            params,
            &mut residuals,
            &mut forecasts,
            &mut amse,
            &mut denom,
            true,
        );
        assert_approx_eq::assert_approx_eq!(fit.lik, 2070.2270304137766);
        assert_approx_eq::assert_approx_eq!(amse[0], 12170.41518101);
        assert_approx_eq::assert_approx_eq!(amse[1], 12649.04373164);
        assert_approx_eq::assert_approx_eq!(amse[2], 13109.83417796);
    }
}
