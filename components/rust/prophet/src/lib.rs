#[allow(warnings)]
#[rustfmt::skip]
mod bindings;
mod conversions;
mod stan_optimizer;

use std::cell::RefCell;

use bindings::exports::augurs::prophet::prophet::{
    Guest, GuestProphet, OptimizeOpts, PredictionData, Predictions, Prophet, ProphetOpts,
    TrainingData,
};
use stan_optimizer::StanOptimizer;

#[allow(clippy::derivable_impls)]
impl Default for OptimizeOpts {
    fn default() -> Self {
        Self {
            algorithm: None,
            seed: None,
            chain: None,
            init_alpha: None,
            tol_obj: None,
            tol_rel_obj: None,
            tol_grad: None,
            tol_rel_grad: None,
            tol_param: None,
            history_size: None,
            iter: None,
            jacobian: None,
            refresh: None,
        }
    }
}

struct Component;

/// Implementation of the `prophet` resource from the `prophet.wit` file.
struct ProphetWrapper {
    /// The inner Prophet model.
    // Trait methods created by `wit-bindgen` do not take `self`
    // by mutable reference, so we can't modify the inner Prophet
    // struct unless we wrap it in a `RefCell`, which gives us
    // interior mutability.
    inner: RefCell<augurs_prophet::Prophet<StanOptimizer>>,
}

impl GuestProphet for ProphetWrapper {
    // /// Construct a Prophet model.
    fn new(opts: Option<ProphetOpts>) -> Result<Prophet, String> {
        let optimizer = StanOptimizer;
        let prophet = Self {
            inner: RefCell::new(augurs_prophet::Prophet::new(
                opts.map(augurs_prophet::ProphetOptions::try_from)
                    .transpose()
                    .map_err(|e| e.to_string())?
                    .unwrap_or_default(),
                optimizer,
            )),
        };
        Ok(Prophet::new(prophet))
    }

    /// Fit the Prophet model to the given data.
    fn fit(&self, data: TrainingData, opts: Option<OptimizeOpts>) -> Result<(), String> {
        let data = augurs_prophet::TrainingData::try_from(data).map_err(|e| e.to_string())?;
        self.inner
            .borrow_mut()
            .fit(data, opts.unwrap_or_default().into())
            .map_err(|e| e.to_string())
    }

    /// Predict future values.
    fn predict(&self, data: Option<PredictionData>) -> Result<Predictions, String> {
        let data = data
            .map(augurs_prophet::PredictionData::try_from)
            .transpose()
            .map_err(|e| e.to_string())?;
        self.inner
            .borrow()
            .predict(data)
            .map(Into::into)
            .map_err(|e| e.to_string())
    }
}

// Implementation of the `model` interface from the `prophet.wit` file.
impl Guest for Component {
    type Prophet = ProphetWrapper;
}

bindings::export!(Component with_types_in bindings);
