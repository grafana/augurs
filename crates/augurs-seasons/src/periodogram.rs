use std::cmp::Ordering;

use itertools::Itertools;
use welch_sde::{Build, SpectralDensity};

// Default number of cycles of data assumed when establishing FFT window sizes.
const DEFAULT_MIN_FFT_CYCLES: f64 = 3.0;

/// Default maximum period assumed when establishing FFT window sizes.
const DEFAULT_MAX_FFT_PERIOD: f64 = 512.0;

/// A builder for a periodogram detector.
#[derive(Debug, Clone)]
pub struct Builder {
    min_period: u32,
    max_period: Option<u32>,
    threshold: f64,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            min_period: 4,
            max_period: None,
            threshold: 0.9,
        }
    }
}

impl Builder {
    /// Set the minimum period to consider when detecting seasonal periods.
    ///
    /// The default is 4.
    #[must_use]
    pub fn min_period(mut self, min_period: u32) -> Self {
        self.min_period = min_period;
        self
    }

    /// Set the maximum period to consider when detecting seasonal periods.
    ///
    /// The default is the length of the data divided by 3, or 512, whichever is smaller.
    #[must_use]
    pub fn max_period(mut self, max_period: u32) -> Self {
        self.max_period = Some(max_period);
        self
    }

    /// Set the threshold for detecting peaks in the periodogram.
    ///
    /// The value will be clamped to the range 0.01 to 0.99.
    ///
    /// The default is 0.9.
    #[must_use]
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.01, 0.99);
        self
    }

    /// Build the periodogram detector.
    ///
    /// The data is the time series to detect seasonal periods in.
    #[must_use]
    pub fn build(self) -> Detector {
        Detector {
            min_period: self.min_period,
            max_period: self.max_period,
            threshold: self.threshold,
        }
    }
}

fn default_max_period(data: &[f64]) -> u32 {
    (data.len() as f64 / DEFAULT_MIN_FFT_CYCLES).min(DEFAULT_MAX_FFT_PERIOD) as u32
}

/// A periodogram of a time series.
#[derive(Debug, Clone, PartialEq)]
pub struct Periodogram {
    /// The periods of the periodogram.
    pub periods: Vec<u32>,
    /// The powers of the periodogram.
    pub powers: Vec<f64>,
}

impl Periodogram {
    /// Find the peaks in the periodogram.
    ///
    /// The peaks are defined as the periods which have a power greater than `threshold` times the
    /// maximum power in the periodogram.
    pub fn peaks(&self, threshold: f64) -> impl Iterator<Item = Period> {
        // Scale the threshold so that it's relative to the maximum power.
        let keep = self
            .powers
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(1.0)
            * threshold;

        // We're going to window by 3 and zip this up with the powers, but we want the
        // middle element to represent the periodogram value we're looking at, so
        // we need to prepend and append a 0 to the periods.
        std::iter::once(0)
            .chain(self.periods.iter().copied())
            .chain(std::iter::once(0))
            .tuple_windows()
            .zip(self.powers.iter().copied())
            .filter_map(|((prev_period, period, next_period), power)| {
                (power >= keep).then_some(Period {
                    power,
                    period,
                    prev_period,
                    next_period,
                })
            })
            .sorted_by(|a, b| a.power.partial_cmp(&b.power).unwrap_or(Ordering::Equal))
    }
}

/// A peak in the periodogram.
#[derive(Debug, Clone, PartialEq)]
pub struct Period {
    /// The power of the peak.
    pub power: f64,
    /// The period of the peak.
    pub period: u32,
    /// The previous period in the periodogram.
    pub prev_period: u32,
    /// The next period in the periodogram.
    pub next_period: u32,
}

/// A season detector which uses a periodogram to identify seasonal periods.
///
/// The detector works by calculating a robust periodogram of the data using
/// Welch's method. The peaks in the periodogram represent likely seasonal periods
/// in the data.
#[derive(Debug)]
pub struct Detector {
    min_period: u32,
    max_period: Option<u32>,
    threshold: f64,
}

impl Detector {
    /// Create a new detector builder.
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Calculate the periodogram of the data.
    ///
    /// The periodogram is a frequency domain representation of the data, and is calculated using the
    /// Welch method.
    ///
    /// The periodogram can then be used to identify peaks, which are returned as periods which
    /// correspond to likely seasonal periods in the data.
    #[must_use]
    pub fn periodogram(&self, data: &[f64]) -> Periodogram {
        let max_period = self.max_period.unwrap_or_else(|| default_max_period(data));
        let frequency = 1.0;
        let data_len = data.len();
        let n_per_segment = (max_period * 2).min(data_len as u32 / 2);
        let max_fft_size = (n_per_segment as f64).log2().floor() as usize;
        let n_segments = (data_len as f64 / n_per_segment as f64).ceil() as usize;

        let welch: SpectralDensity<'_, f64> = SpectralDensity::builder(data, frequency)
            .n_segment(n_segments)
            .dft_log2_max_size(max_fft_size)
            .build();
        let sd = welch.periodogram();

        let freqs = sd.frequency();
        // Periods are the reciprocal of the frequency, since we've used a frequency of 1.
        // Make sure we skip the first one, which is 0, and the first power, which corresponds to
        // that.
        let periods = freqs.iter().skip(1).map(|x| x.recip().round() as u32);
        let power = sd.iter().skip(1).copied();

        let (periods, powers) = periods
            .zip(power)
            .filter(|(per, _)| {
                // Filter out periods that are too short or too long, and the period corresponding to the
                // segment length.
                *per >= self.min_period && *per < max_period && *per != n_per_segment
            })
            // Group by period, and keep the maximum power for each period.
            .group_by(|(per, _)| *per)
            .into_iter()
            .map(|(per, group)| {
                let max_power = group
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or((0, 0.0));
                (per, max_power.1)
            })
            .unzip();
        Periodogram { periods, powers }
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl crate::Detector for Detector {
    fn detect(&self, data: &[f64]) -> Vec<u32> {
        self.periodogram(data)
            .peaks(self.threshold)
            .map(|x| x.period)
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{test_data::*, Detector as _};

    #[test]
    fn smoke() {
        #[rustfmt::skip]
        let y = &[
            0.1, 0.3, 0.8, 0.5,
            0.1, 0.31, 0.79, 0.48,
            0.09, 0.29, 0.81, 0.49,
            0.11, 0.28, 0.78, 0.53,
            0.1, 0.3, 0.8, 0.5,
            0.1, 0.31, 0.79, 0.48,
            0.09, 0.29, 0.81, 0.49,
            0.11, 0.28, 0.78, 0.53,
        ];
        let periods = Detector::default().detect(y);
        assert_eq!(periods[0], 4);
    }

    #[test]
    fn test_detect() {
        for (i, test_case) in CASES.iter().enumerate() {
            let TestCase {
                data,
                season_lengths: expected,
            } = test_case;
            let detector = Detector::default();
            assert_eq!(
                detector
                    .periodogram(data)
                    .peaks(0.5)
                    .map(|x| x.period)
                    .collect_vec(),
                *expected,
                "Test case {}",
                i
            );
        }
    }
}
