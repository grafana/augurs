use std::collections::HashMap;

use crate::{Error, TimestampSeconds};

/// The data needed to train a Prophet model.
///
/// Create a `TrainingData` object with the `new` method, which
/// takes a vector of dates and a vector of values.
///
/// Optionally, you can add seasonality conditions, regressors,
/// floor and cap columns.
#[derive(Clone, Debug)]
pub struct TrainingData {
    pub(crate) n: usize,
    pub(crate) ds: Vec<TimestampSeconds>,
    pub(crate) y: Vec<f64>,
    pub(crate) cap: Option<Vec<f64>>,
    pub(crate) floor: Option<Vec<f64>>,
    pub(crate) seasonality_conditions: HashMap<String, Vec<bool>>,
    pub(crate) x: HashMap<String, Vec<f64>>,
}

impl TrainingData {
    /// Create some input data for Prophet.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `ds` and `y` differ.
    pub fn new(ds: Vec<TimestampSeconds>, y: Vec<f64>) -> Result<Self, Error> {
        if ds.len() != y.len() {
            return Err(Error::MismatchedLengths {
                a_name: "ds".to_string(),
                a: ds.len(),
                b_name: "y".to_string(),
                b: y.len(),
            });
        }
        Ok(Self {
            n: ds.len(),
            ds,
            y,
            cap: None,
            floor: None,
            seasonality_conditions: HashMap::new(),
            x: HashMap::new(),
        })
    }

    /// Add the cap for logistic growth.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `ds` and `cap` differ.
    pub fn with_cap(mut self, cap: Vec<f64>) -> Result<Self, Error> {
        if self.n != cap.len() {
            return Err(Error::MismatchedLengths {
                a_name: "ds".to_string(),
                a: self.ds.len(),
                b_name: "cap".to_string(),
                b: cap.len(),
            });
        }
        self.cap = Some(cap);
        Ok(self)
    }

    /// Add the floor for logistic growth.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `ds` and `floor` differ.
    pub fn with_floor(mut self, floor: Vec<f64>) -> Result<Self, Error> {
        if self.n != floor.len() {
            return Err(Error::MismatchedLengths {
                a_name: "ds".to_string(),
                a: self.ds.len(),
                b_name: "floor".to_string(),
                b: floor.len(),
            });
        }
        self.floor = Some(floor);
        Ok(self)
    }

    /// Add condition columns for conditional seasonalities.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `ds` and any of the seasonality
    /// condition columns differ.
    pub fn with_seasonality_conditions(
        mut self,
        seasonality_conditions: HashMap<String, Vec<bool>>,
    ) -> Result<Self, Error> {
        for (name, cond) in seasonality_conditions.iter() {
            if self.n != cond.len() {
                return Err(Error::MismatchedLengths {
                    a_name: "ds".to_string(),
                    a: self.ds.len(),
                    b_name: name.clone(),
                    b: cond.len(),
                });
            }
        }
        self.seasonality_conditions = seasonality_conditions;
        Ok(self)
    }

    /// Add regressors.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `ds` and any of the regressor
    /// columns differ.
    pub fn with_regressors(mut self, x: HashMap<String, Vec<f64>>) -> Result<Self, Error> {
        for (name, reg) in x.iter() {
            if self.n != reg.len() {
                return Err(Error::MismatchedLengths {
                    a_name: "ds".to_string(),
                    a: self.ds.len(),
                    b_name: name.clone(),
                    b: reg.len(),
                });
            }
            if reg.iter().any(|x| x.is_nan()) {
                return Err(Error::NaNValue {
                    column: name.clone(),
                });
            }
        }
        self.x = x;
        Ok(self)
    }

    #[cfg(test)]
    pub(crate) fn head(mut self, n: usize) -> Self {
        self.n = n;
        self.ds.truncate(n);
        self.y.truncate(n);
        if let Some(cap) = self.cap.as_mut() {
            cap.truncate(n);
        }
        if let Some(floor) = self.floor.as_mut() {
            floor.truncate(n);
        }
        for (_, v) in self.x.iter_mut() {
            v.truncate(n);
        }
        for (_, v) in self.seasonality_conditions.iter_mut() {
            v.truncate(n);
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn tail(mut self, n: usize) -> Self {
        let split = self.ds.len() - n;
        self.n = n;
        self.ds = self.ds.split_off(split);
        self.y = self.y.split_off(split);
        if let Some(cap) = self.cap.as_mut() {
            *cap = cap.split_off(split);
        }
        if let Some(floor) = self.floor.as_mut() {
            *floor = floor.split_off(split);
        }
        for (_, v) in self.x.iter_mut() {
            *v = v.split_off(split);
        }
        for (_, v) in self.seasonality_conditions.iter_mut() {
            *v = v.split_off(split);
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.n
    }
}

/// The data needed to predict with a Prophet model.
///
/// The structure of the prediction data must be the same as the
/// training data used to train the model, with the exception of
/// `y` (which is being predicted).
///
/// That is, if your model used certain seasonality conditions or
/// regressors, you must include them in the prediction data.
#[derive(Clone, Debug)]
pub struct PredictionData {
    pub(crate) n: usize,
    pub(crate) ds: Vec<TimestampSeconds>,
    pub(crate) cap: Option<Vec<f64>>,
    pub(crate) floor: Option<Vec<f64>>,
    pub(crate) seasonality_conditions: HashMap<String, Vec<bool>>,
    pub(crate) x: HashMap<String, Vec<f64>>,
}

impl PredictionData {
    /// Create some data to be used for predictions.
    ///
    /// Predictions will be made for each of the dates in `ds`.
    pub fn new(ds: Vec<TimestampSeconds>) -> Self {
        Self {
            n: ds.len(),
            ds,
            cap: None,
            floor: None,
            seasonality_conditions: HashMap::new(),
            x: HashMap::new(),
        }
    }

    /// Add the cap for logistic growth.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `ds` and `cap` are not equal.
    pub fn with_cap(mut self, cap: Vec<f64>) -> Result<Self, Error> {
        if self.n != cap.len() {
            return Err(Error::MismatchedLengths {
                a_name: "ds".to_string(),
                a: self.ds.len(),
                b_name: "cap".to_string(),
                b: cap.len(),
            });
        }
        self.cap = Some(cap);
        Ok(self)
    }

    /// Add the floor for logistic growth.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `ds` and `floor` are not equal.
    pub fn with_floor(mut self, floor: Vec<f64>) -> Result<Self, Error> {
        if self.n != floor.len() {
            return Err(Error::MismatchedLengths {
                a_name: "ds".to_string(),
                a: self.ds.len(),
                b_name: "floor".to_string(),
                b: floor.len(),
            });
        }
        self.floor = Some(floor);
        Ok(self)
    }

    /// Add condition columns for conditional seasonalities.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of any of the seasonality conditions
    /// are not equal to the length of `ds`.
    pub fn with_seasonality_conditions(
        mut self,
        seasonality_conditions: HashMap<String, Vec<bool>>,
    ) -> Result<Self, Error> {
        for (name, cond) in seasonality_conditions.iter() {
            if self.n != cond.len() {
                return Err(Error::MismatchedLengths {
                    a_name: "ds".to_string(),
                    a: self.ds.len(),
                    b_name: name.clone(),
                    b: cond.len(),
                });
            }
        }
        self.seasonality_conditions = seasonality_conditions;
        Ok(self)
    }

    /// Add regressors.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of any of the regressors
    /// are not equal to the length of `ds`.
    pub fn with_regressors(mut self, x: HashMap<String, Vec<f64>>) -> Result<Self, Error> {
        for (name, reg) in x.iter() {
            if self.n != reg.len() {
                return Err(Error::MismatchedLengths {
                    a_name: "ds".to_string(),
                    a: self.ds.len(),
                    b_name: name.clone(),
                    b: reg.len(),
                });
            }
            if reg.iter().any(|x| x.is_nan()) {
                return Err(Error::NaNValue {
                    column: name.clone(),
                });
            }
        }
        self.x = x;
        Ok(self)
    }
}
