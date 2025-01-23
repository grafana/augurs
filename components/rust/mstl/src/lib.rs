#[allow(warnings)]
#[rustfmt::skip]
mod bindings;

use augurs_core::{Fit, Predict};
use bindings::{
    augurs::core::types::{Error, FittedTrendModel, Forecast, ForecastIntervals, TrendModel},
    exports::augurs::mstl::mstl::{FittedMstl, Guest, GuestFittedMstl, GuestMstl},
};

struct Component;

impl Guest for Component {
    type Mstl = Mstl;
    type FittedMstl = FittedMstlWrapper;
}

#[derive(Debug)]
struct TrendModelWrapper {
    inner: TrendModel,
}
impl TrendModelWrapper {
    fn new_boxed(trend_model: TrendModel) -> Box<dyn augurs_mstl::TrendModel + Send + Sync> {
        Box::new(Self { inner: trend_model })
    }
}

impl augurs_mstl::TrendModel for TrendModelWrapper {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "wasm-component".into()
    }

    fn fit(
        &self,
        y: &[f64],
    ) -> Result<
        Box<dyn augurs_mstl::FittedTrendModel + Sync + Send>,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    > {
        self.inner
            .fit(y)
            .map(|model| FittedTrendModelWrapper::new_boxed(model, Some(y.len())))
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync + 'static>)
    }
}

#[derive(Debug)]
struct FittedTrendModelWrapper {
    inner: FittedTrendModel,
    training_data_size: Option<usize>,
}

impl FittedTrendModelWrapper {
    fn new_boxed(
        inner: FittedTrendModel,
        training_data_size: Option<usize>,
    ) -> Box<dyn augurs_mstl::FittedTrendModel + Send + Sync> {
        Box::new(Self {
            inner,
            training_data_size,
        })
    }
}

impl augurs_mstl::FittedTrendModel for FittedTrendModelWrapper {
    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut augurs_core::Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        let f: augurs_core::Forecast = self.inner.predict(horizon as u32, level)?.into();
        *forecast = f;
        Ok(())
    }

    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut augurs_core::Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        let f: augurs_core::Forecast = self.inner.predict_in_sample(level)?.into();
        *forecast = f;
        Ok(())
    }

    fn training_data_size(&self) -> Option<usize> {
        self.training_data_size
    }
}

struct Mstl {
    inner: augurs_mstl::MSTLModel<Box<dyn augurs_mstl::TrendModel + Send + Sync>>,
}

impl GuestMstl for Mstl {
    fn new(periods: Vec<u32>, trend_model: TrendModel) -> Self {
        let periods = periods.into_iter().map(|x| x as usize).collect();
        Self {
            inner: augurs_mstl::MSTLModel::new(periods, TrendModelWrapper::new_boxed(trend_model)),
        }
    }

    fn fit(&self, y: Vec<f64>) -> Result<FittedMstl, Error> {
        self.inner
            .fit(&y)
            .map(|model| FittedMstl::new(FittedMstlWrapper { inner: model }))
            .map_err(|e| Error::Fit(e.to_string()))
    }
}

struct FittedMstlWrapper {
    inner: augurs_mstl::FittedMSTLModel,
}

impl GuestFittedMstl for FittedMstlWrapper {
    fn predict(&self, horizon: u32, level: Option<f64>) -> Result<Forecast, Error> {
        self.inner
            .predict(horizon as usize, level)
            .map(|f| f.into())
            .map_err(|e| Error::Predict(e.to_string()))
    }
}

impl From<augurs_core::Forecast> for Forecast {
    fn from(f: augurs_core::Forecast) -> Self {
        Self {
            point: f.point,
            intervals: f.intervals.map(|i| i.into()),
        }
    }
}

impl From<augurs_core::ForecastIntervals> for ForecastIntervals {
    fn from(f: augurs_core::ForecastIntervals) -> Self {
        Self {
            level: f.level,
            lower: f.lower,
            upper: f.upper,
        }
    }
}

impl From<Forecast> for augurs_core::Forecast {
    fn from(f: Forecast) -> Self {
        Self {
            point: f.point,
            intervals: f.intervals.map(|i| i.into()),
        }
    }
}

impl From<ForecastIntervals> for augurs_core::ForecastIntervals {
    fn from(f: ForecastIntervals) -> Self {
        Self {
            level: f.level,
            lower: f.lower,
            upper: f.upper,
        }
    }
}

bindings::export!(Component with_types_in bindings);
