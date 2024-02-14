use augurs_testing::data::{SEASON_EIGHT, SEASON_SEVEN};

pub(crate) struct TestCase {
    pub(crate) season_lengths: &'static [u32],
    pub(crate) data: &'static [f64],
}

pub(crate) static CASES: &[TestCase] = &[
    TestCase {
        season_lengths: &[8],
        data: SEASON_EIGHT,
    },
    TestCase {
        season_lengths: &[7],
        data: SEASON_SEVEN,
    },
];
