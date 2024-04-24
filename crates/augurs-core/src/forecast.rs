/// Forecast intervals.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForecastIntervals {
    /// The confidence level for the intervals.
    pub level: f64,
    /// The lower prediction intervals.
    pub lower: Vec<f64>,
    /// The upper prediction intervals.
    pub upper: Vec<f64>,
}

impl ForecastIntervals {
    /// Return empty forecast intervals.
    pub fn empty(level: f64) -> ForecastIntervals {
        Self {
            level,
            lower: Vec::new(),
            upper: Vec::new(),
        }
    }

    /// Return empty forecast intervals with the specified capacity.
    pub fn with_capacity(level: f64, capacity: usize) -> ForecastIntervals {
        Self {
            level,
            lower: Vec::with_capacity(capacity),
            upper: Vec::with_capacity(capacity),
        }
    }
}

/// A forecast containing point forecasts and, optionally, prediction intervals.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Forecast {
    /// The point forecasts.
    pub point: Vec<f64>,
    /// The forecast intervals, if requested and supported
    /// by the trend model.
    pub intervals: Option<ForecastIntervals>,
}

impl Forecast {
    /// Return an empty forecast.
    pub fn empty() -> Forecast {
        Self {
            point: Vec::new(),
            intervals: None,
        }
    }

    /// Return an empty forecast with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Forecast {
        Self {
            point: Vec::with_capacity(capacity),
            intervals: None,
        }
    }

    /// Return an empty forecast with the specified capacity and level.
    pub fn with_capacity_and_level(capacity: usize, level: f64) -> Forecast {
        Self {
            point: Vec::with_capacity(capacity),
            intervals: Some(ForecastIntervals::with_capacity(level, capacity)),
        }
    }
}
