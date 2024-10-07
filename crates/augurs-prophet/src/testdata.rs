use std::{fs, path::Path, str::FromStr};

use chrono::{DateTime, NaiveDate, Utc};

use crate::{TimestampSeconds, TrainingData};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data");

enum Ds {
    Date(NaiveDate),
    DateTime(DateTime<Utc>),
}
impl Ds {
    fn timestamp(&self) -> u64 {
        (match self {
            Ds::Date(d) => d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp(),
            Ds::DateTime(dt) => dt.timestamp(),
        }) as u64
    }
}

impl FromStr for Ds {
    type Err = chrono::ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.len() {
            10 => Ds::Date(s.parse()?),
            _ => Ds::DateTime(s.parse()?),
        })
    }
}

fn load_csv(path: &str) -> (Vec<TimestampSeconds>, Vec<f64>) {
    let path = Path::new(DATA_DIR).join(path);
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| -> (TimestampSeconds, f64) {
            let mut parts = line.split(',');
            let ds = parts.next().unwrap().parse::<Ds>().unwrap();
            let y = parts.next().unwrap().parse().unwrap();
            (ds.timestamp(), y)
        })
        .unzip()
}

pub(crate) fn daily_univariate_ts() -> TrainingData {
    let (ds, y) = load_csv("daily_univariate_ts.csv");
    TrainingData::new(ds, y).unwrap()
}

pub(crate) fn train_test_split(data: TrainingData, ratio: f64) -> (TrainingData, TrainingData) {
    let n = data.len();
    let split = (n as f64 * ratio).round() as usize;
    let test_size = n - split;
    train_test_splitn(data, test_size)
}

pub(crate) fn train_test_splitn(
    data: TrainingData,
    test_size: usize,
) -> (TrainingData, TrainingData) {
    let n = data.len();
    let train_size = n - test_size;
    (data.clone().head(train_size), data.tail(test_size))
}
