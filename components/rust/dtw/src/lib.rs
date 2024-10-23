#[allow(warnings)]
#[rustfmt::skip]
mod bindings;

use bindings::exports::augurs::dtw::dtw::{Dtw, DtwOpts, Guest, GuestDtw, Series};

struct Component;

impl Guest for Component {
    type Dtw = DtwWrapper;
}

enum DtwWrapper {
    Euclidean(augurs_dtw::Dtw<augurs_dtw::Euclidean>),
    Manhattan(augurs_dtw::Dtw<augurs_dtw::Manhattan>),
}

impl GuestDtw for DtwWrapper {
    fn euclidean(opts: Option<DtwOpts>) -> Dtw {
        let mut dtw = augurs_dtw::Dtw::euclidean();
        if let Some(window) = opts.map(|x| x.window) {
            dtw = dtw.with_window(window as usize);
        }
        Dtw::new(Self::Euclidean(dtw))
    }

    fn manhattan(opts: Option<DtwOpts>) -> Dtw {
        let mut dtw = augurs_dtw::Dtw::manhattan();
        if let Some(window) = opts.map(|x| x.window) {
            dtw = dtw.with_window(window as usize);
        }
        Dtw::new(Self::Manhattan(dtw))
    }

    fn distance(&self, s: Series, t: Series) -> f64 {
        match self {
            Self::Euclidean(dtw) => dtw.distance(&s, &t),
            Self::Manhattan(dtw) => dtw.distance(&s, &t),
        }
    }

    fn distance_matrix(&self, series: Vec<Series>) -> Vec<Vec<f64>> {
        let series_slice = series.iter().map(Vec::as_slice).collect::<Vec<_>>();
        match self {
            Self::Euclidean(dtw) => dtw.distance_matrix(&series_slice).into_inner(),
            Self::Manhattan(dtw) => dtw.distance_matrix(&series_slice).into_inner(),
        }
    }
}

bindings::export!(Component with_types_in bindings);
