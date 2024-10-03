use std::collections::HashMap;

use crate::TimestampSeconds;

/// The data needed to train a Prophet model.
///
/// Create a `TrainingData` object with the `new` method, which
/// takes a vector of dates and a vector of values.
///
/// Optionally, you can add seasonality indicators, regressors,
/// floor and cap columns.
pub struct TrainingData {
    pub(crate) ds: Vec<TimestampSeconds>,
    pub(crate) y: Vec<f64>,
    pub(crate) seasonal_indicators: HashMap<String, Vec<bool>>,
    pub(crate) x: HashMap<String, Vec<f64>>,
    pub(crate) floor: Option<Vec<f64>>,
    pub(crate) cap: Option<Vec<f64>>,
}

impl TrainingData {
    /// Create some input data for Prophet.
    pub fn new(ds: Vec<TimestampSeconds>, y: Vec<f64>) -> Self {
        Self {
            ds,
            y,
            seasonal_indicators: HashMap::new(),
            x: HashMap::new(),
            floor: None,
            cap: None,
        }
    }

    /// Add indicators for conditional seasonalities.
    pub fn with_seasonality_indicators(
        mut self,
        seasonal_indicators: HashMap<String, Vec<bool>>,
    ) -> Self {
        self.seasonal_indicators = seasonal_indicators;
        self
    }

    /// Add regressors.
    pub fn with_regressors(mut self, x: HashMap<String, Vec<f64>>) -> Self {
        self.x = x;
        self
    }

    /// Add the floor for logistic growth.
    pub fn with_floor(mut self, floor: Vec<f64>) -> Self {
        self.floor = Some(floor);
        self
    }

    /// Add the cap for logistic growth.
    pub fn with_cap(mut self, cap: Vec<f64>) -> Self {
        self.cap = Some(cap);
        self
    }
}
