//! Integration tests for the augurs wrapper crate.

#[cfg(feature = "changepoint")]
#[test]
fn test_changepoint() {
    use augurs::changepoint::{ArgpcpDetector, Detector};
    let data = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    let changepoints = ArgpcpDetector::builder().build().detect_changepoints(&data);
    // 1 changepoint, but the start is considered a changepoint too.
    assert_eq!(changepoints, vec![0, 33]);
}

#[cfg(feature = "clustering")]
#[test]
fn test_clustering() {
    use augurs::{clustering::DbscanClusterer, DistanceMatrix};
    let distance_matrix = vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 0.0, 3.0, 3.0],
        vec![2.0, 3.0, 0.0, 4.0],
        vec![3.0, 3.0, 4.0, 0.0],
    ];
    let distance_matrix = DistanceMatrix::try_from_square(distance_matrix).unwrap();
    let clusters = DbscanClusterer::new(0.5, 2).fit(&distance_matrix);
    assert_eq!(clusters, vec![-1, -1, -1, -1]);

    let clusters = DbscanClusterer::new(1.0, 2).fit(&distance_matrix);
    assert_eq!(clusters, vec![0, 0, -1, -1]);

    let clusters = DbscanClusterer::new(1.0, 3).fit(&distance_matrix);
    assert_eq!(clusters, vec![-1, -1, -1, -1]);

    let clusters = DbscanClusterer::new(2.0, 2).fit(&distance_matrix);
    assert_eq!(clusters, vec![0, 0, 0, -1]);

    let clusters = DbscanClusterer::new(2.0, 3).fit(&distance_matrix);
    assert_eq!(clusters, vec![0, 0, 0, -1]);

    let clusters = DbscanClusterer::new(3.0, 3).fit(&distance_matrix);
    assert_eq!(clusters, vec![0, 0, 0, 0]);
}

#[cfg(feature = "dtw")]
#[test]
fn test_dtw() {
    use augurs::dtw::Dtw;
    use augurs_testing::assert_approx_eq;
    let result = Dtw::euclidean().distance(&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0]);
    assert_approx_eq!(result, 5.0990195135927845);
}

#[cfg(feature = "dtw")]
#[test]
fn test_dtw_distance_matrix() {
    use augurs::dtw::Dtw;
    use augurs_testing::assert_all_close;

    let dtw = Dtw::euclidean();
    let series: &[&[f64]] = &[&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0], &[6.0, 7.0, 8.0]];
    let dists = dtw.distance_matrix(series);
    assert_eq!(dists.shape(), (3, 3));
    assert_all_close(&dists[0], &[0.0, 5.0990195135927845, 10.392304845413264]);

    // Test with different length series.
    let dtw = Dtw::euclidean();
    let series: &[&[f64]] = &[&[0.0, 1.0, 2.0], &[3.0], &[6.0, 7.0]];
    let dists = dtw.distance_matrix(series);
    assert_eq!(dists.shape(), (3, 3));
    assert_all_close(&dists[0], &[0.0, 3.7416573867739413, 9.273618495495704]);
}

#[cfg(feature = "ets")]
#[test]
fn test_ets() {
    use augurs::{
        ets::{
            model::{ErrorComponent, SeasonalComponent, TrendComponent},
            AutoETS,
        },
        prelude::*,
    };
    use augurs_testing::{assert_within_pct, data::AIR_PASSENGERS};

    let auto = AutoETS::non_seasonal();
    let fit = auto.fit(AIR_PASSENGERS).expect("fit failed");
    assert_eq!(
        fit.model().model_type().error,
        ErrorComponent::Multiplicative
    );
    assert_eq!(fit.model().model_type().trend, TrendComponent::Additive);
    assert_eq!(fit.model().model_type().season, SeasonalComponent::None);
    assert_within_pct!(fit.model().log_likelihood(), -831.4883541595792, 0.01);
    assert_within_pct!(fit.model().aic(), 1672.9767083191584, 0.01);
}

#[cfg(feature = "forecaster")]
#[test]
fn test_forecaster() {
    use augurs::{
        forecaster::{transforms::MinMaxScaler, Forecaster, Transformer},
        mstl::{MSTLModel, NaiveTrend},
    };
    use augurs_forecaster::transforms::{LinearInterpolator, Logit};
    use augurs_testing::{assert_all_close, data::AIR_PASSENGERS};

    let transforms = vec![
        LinearInterpolator::new().boxed(),
        MinMaxScaler::new().boxed(),
        Logit::new().boxed(),
    ];
    let model = MSTLModel::new(vec![2], NaiveTrend::new());
    let mut forecaster = Forecaster::new(model).with_transformers(transforms);
    forecaster.fit(AIR_PASSENGERS).unwrap();
    let forecasts = forecaster.predict(4, None).unwrap();
    dbg!(&forecasts.point);
    assert_all_close(
        &forecasts.point,
        &[
            620.5523022842495,
            431.9999972537765,
            620.5523022842495,
            431.9999972537765,
        ],
    );
}

#[cfg(feature = "mstl")]
#[test]
fn test_mstl() {
    use augurs::{
        mstl::{stlrs, MSTLModel, NaiveTrend},
        prelude::*,
    };
    use augurs_testing::{assert_all_close, data::VIC_ELEC};

    let mut stl_params = stlrs::params();
    stl_params
        .seasonal_degree(0)
        .seasonal_jump(1)
        .trend_degree(1)
        .trend_jump(1)
        .low_pass_degree(1)
        .inner_loops(2)
        .outer_loops(0);
    let mut mstl_params = stlrs::MstlParams::new();
    mstl_params.stl_params(stl_params);
    let periods = vec![24, 24 * 7];
    let trend_model = NaiveTrend::new();
    let mstl = MSTLModel::new(periods, trend_model).mstl_params(mstl_params);
    let fit = mstl.fit(&VIC_ELEC).unwrap();

    let in_sample = fit.predict_in_sample(0.95).unwrap();
    // The first 12 values from R.
    let expected_in_sample = vec![
        f64::NAN,
        7952.216,
        7269.439,
        6878.110,
        6606.999,
        6402.581,
        6659.523,
        7457.488,
        8111.359,
        8693.762,
        9255.807,
        9870.213,
    ];
    assert_eq!(in_sample.point.len(), VIC_ELEC.len());
    assert_all_close(&in_sample.point[..12], &expected_in_sample);

    let out_of_sample = fit.predict(10, 0.95).unwrap();
    let expected_out_of_sample: Vec<f64> = vec![
        8920.670, 8874.234, 8215.508, 7782.726, 7697.259, 8216.241, 9664.907, 10914.452, 11536.929,
        11664.737,
    ];
    let expected_out_of_sample_lower = vec![
        8700.984, 8563.551, 7835.001, 7343.354, 7206.026, 7678.122, 9083.672, 10293.087, 10877.871,
        10970.029,
    ];
    let expected_out_of_sample_upper = vec![
        9140.356, 9184.917, 8596.016, 8222.098, 8188.491, 8754.359, 10246.141, 11535.818,
        12195.987, 12359.445,
    ];
    assert_eq!(out_of_sample.point.len(), 10);
    assert_all_close(&out_of_sample.point, &expected_out_of_sample);
    let ForecastIntervals { lower, upper, .. } = out_of_sample.intervals.unwrap();
    assert_eq!(lower.len(), 10);
    assert_eq!(upper.len(), 10);
    assert_all_close(&lower, &expected_out_of_sample_lower);
    assert_all_close(&upper, &expected_out_of_sample_upper);
}

#[cfg(feature = "outlier")]
#[test]
fn test_outlier_dbscan() {
    use augurs::outlier::{DbscanDetector, OutlierDetector};
    let data: &[&[f64]] = &[
        &[1.0, 2.0, 1.5, 2.3],
        &[1.9, 2.2, 1.2, 2.4],
        &[1.5, 2.1, 6.4, 8.5],
    ];
    let detector =
        DbscanDetector::with_sensitivity(0.5).expect("sensitivity is between 0.0 and 1.0");
    let processed = detector.preprocess(data).unwrap();
    let outliers = detector.detect(&processed).unwrap();

    assert_eq!(outliers.outlying_series.len(), 1);
    assert!(outliers.outlying_series.contains(&2));
    assert!(outliers.series_results[2].is_outlier);
    assert_eq!(outliers.series_results[2].scores, vec![0.0, 0.0, 1.0, 1.0]);
    assert!(outliers.cluster_band.is_some());
}

#[cfg(feature = "outlier")]
#[test]
fn test_outlier_mad() {
    use augurs::outlier::{MADDetector, OutlierDetector};
    let data: &[&[f64]] = &[
        &[1.0, 2.0, 1.5, 2.3],
        &[1.9, 2.2, 1.2, 2.4],
        &[1.5, 2.1, 6.4, 8.5],
    ];
    let detector = MADDetector::with_sensitivity(0.5).unwrap();
    let processed = detector.preprocess(data).unwrap();
    let outliers = detector.detect(&processed).unwrap();

    assert_eq!(outliers.outlying_series.len(), 1);
    assert!(outliers.outlying_series.contains(&2));
    assert!(outliers.series_results[2].is_outlier);
    assert_eq!(
        outliers.series_results[2].scores,
        vec![
            0.6835259767082061,
            0.057793242408848366,
            5.028012089569781,
            7.4553282707414
        ]
    );
    assert!(outliers.cluster_band.is_some());
}

#[cfg(feature = "seasons")]
#[test]
fn test_seasonal() {
    use augurs::seasons::{Detector, PeriodogramDetector};

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
    let periods = PeriodogramDetector::default().detect(y);
    assert_eq!(periods[0], 4);
}
