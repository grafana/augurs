# Migration guide

This guide will help you migrate from the previous version of `augurs` to the latest version.

## From 0.7 to 0.8

### Transformations

In version 0.8 the `augurs::forecaster::Transform` enum was removed and replaced with the
`augurs::forecaster::Transformer` trait, which closely mirrors the scikit-learn `Transformer`
API. The various `Transform` enum variants were replaced with the following `Transformer`
implementations, such as `augurs::forecaster::transforms::MinMaxScaler`. The new `Pipeline`
struct is a `Transformer` implementation that can be used to chain multiple transformations
together.

Whereas some transformations previously needed to be passed the data in the constructor, this
is now handled by the `fit` method of the `Transformer` trait, allowing the transformations
to be more lazy.

It also makes it possible to implement custom transformations by implementing the `Transformer`
trait.

Before:

```rust,ignore
# extern crate augurs;
use augurs::{
    forecaster::{transforms::MinMaxScaleParams, Forecaster, Transform},
    mstl::{MSTLModel, NaiveTrend},
};

let transforms = vec![
    Transform::linear_interpolator(),
    Transform::min_max_scaler(MinMaxScaleParams::new(0.0, 1.0)),
    Transform::log(),
];
// use the transforms in a forecaster:
let model = MSTLModel::new(vec![2], NaiveTrend::new());
let mut forecaster = Forecaster::new(model).with_transforms(transforms);
```

After:

```rust
# extern crate augurs;
use augurs::{
    forecaster::{
        transforms::{LinearInterpolator, Log, MinMaxScaler},
        Forecaster, Transformer,
    },
    mstl::{MSTLModel, NaiveTrend},
};

let transformers = vec![
    LinearInterpolator::new().boxed(),
    MinMaxScaler::new().with_scaled_range(0.0, 1.0).boxed(),
    Log::new().boxed(),
];
// use the transformers in a forecaster:
let model = MSTLModel::new(vec![2], NaiveTrend::new());
let mut forecaster = Forecaster::new(model).with_transformers(transformers);
```

### Prophet

In version 0.8 the `augurs::prophet::Prophet` struct was removed and replaced with the
`augurs::prophet::ProphetForecaster` struct, which is a `Forecaster` implementation that
uses the Prophet model to forecast future values.

## From 0.6 to 0.7

### Prophet

Version 0.7 made changes to the way that holidays are treated in the Prophet model ([PR #181](https://github.com/grafana/augurs/pull/181)).

In versions prior to 0.7, holidays were implicitly assumed to last 1 day each, starting and
ending at midnight UTC. This stemmed from how the Python API works: holidays are passed as
a column of dates in a pandas DataFrame.

In version 0.7, each holiday is instead specified using a list of `HolidayOccurrence`s, which
each have a start and end time represented as Unix timestamps. This allows you to specify
holidays more flexibly:

- holidays lasting 1 day from midnight to midnight UTC can be specified using `HolidayOccurrence::for_day`.
  This is the equivalent of the previous behavior.
- holidays lasting 1 day in a non-UTC timezone can be specified using `HolidayOccurrence::for_day_in_tz`.
  The second argument is the offset in seconds from UTC, which can be calculated manually or using
  the `chrono::FixedOffset::local_minus_utc` method.
- holidays lasting for custom periods, such as sub-daily or multi-day periods, can be specified using
  `HolidayOccurrence::new` with a start and end time in seconds since the Unix epoch.

In short, you can replace the following code:

```rust,ignore
# extern crate augurs;
# extern crate chrono;
use augurs::prophet::Holiday;
use chrono::{prelude::*, Utc};

let holiday_date = Utc.with_ymd_and_hms(2022, 6, 12, 0, 0, 0).unwrap().timestamp();
let holiday = Holiday::new(vec![holiday_date]);
```

with the following code:

```rust,ignore
# extern crate augurs;
# extern crate chrono;
use augurs::prophet::{Holiday, HolidayOccurrence};
use chrono::{prelude::*, Utc};

let holiday_date = Utc.with_ymd_and_hms(2022, 6, 12, 0, 0, 0).unwrap().timestamp();
let occurrence = HolidayOccurrence::for_day(holiday_date);
let holiday = Holiday::new(vec![occurrence]);
```

Or use `HolidayOccurrence::for_day_in_tz` to specify a holiday in a non-UTC timezone:

```rust,ignore
# extern crate augurs;
# extern crate chrono;
use augurs::prophet::{Holiday, HolidayOccurrence};
use chrono::{prelude::*, Utc};

let holiday_date = Utc.with_ymd_and_hms(2022, 6, 12, 0, 0, 0).unwrap().timestamp();
// This holiday lasts for 1 day in UTC+1.
let occurrence = HolidayOccurrence::for_day_in_tz(holiday_date, 3600);
let holiday = Holiday::new(vec![occurrence]);
```

Or use `HolidayOccurrence::new` to specify a holiday with a custom start and end time:

```rust,ignore
# extern crate augurs;
# extern crate chrono;
use augurs::prophet::{Holiday, HolidayOccurrence};
use chrono::{prelude::*, Utc};

let holiday_date = Utc.with_ymd_and_hms(2022, 6, 12, 0, 0, 0).unwrap().timestamp();
// This holiday lasts for 1 hour.
let occurrence = HolidayOccurrence::new(holiday_date, holiday_date + 3600);
let holiday = Holiday::new(vec![occurrence]);
```
