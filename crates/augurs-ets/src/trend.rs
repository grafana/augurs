use std::borrow::Cow;

use augurs_core::{Forecast, ForecastIntervals};
use augurs_mstl::TrendModel;

use crate::AutoETS;

impl TrendModel for AutoETS {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("AutoETS")
    }

    fn fit(&mut self, y: &[f64]) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        Ok(self.fit(y).map(|_| ())?)
    }

    fn predict(
        &self,
        horizon: usize,
        level: Option<f64>,
    ) -> Result<Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Ok(self.predict(horizon, level).map(|forecast| Forecast {
            point: forecast.point,
            intervals: forecast.intervals.map(|fi| ForecastIntervals {
                level: fi.level,
                lower: fi.lower,
                upper: fi.upper,
            }),
        })?)
    }

    fn predict_in_sample(
        &self,
        level: Option<f64>,
    ) -> Result<Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Ok(self.predict_in_sample(level).map(|forecast| Forecast {
            point: forecast.point,
            intervals: forecast.intervals.map(|fi| ForecastIntervals {
                level: fi.level,
                lower: fi.lower,
                upper: fi.upper,
            }),
        })?)
    }
}
