# Dynamic Time Warping (DTW)

Dynamic time warping can be used to compare two sequences that may vary in time or speed.

This implementation has built-in support for both Euclidean and Manhattan distance,
and can be extended to support other distance functions by implementing the [`Distance`]
trait and using the [`Dtw::new`] constructor.

The algorithm is based on the code from the [UCR Suite][ucr-suite]. Benchmarks show similar
timings to `dtaidistance`'s C implementation, but note that `dtaidistance` is much more
full featured!

# Features

- [x] DTW distance between two sequences
- [x] optimized scalar implementation influenced by the [UCR Suite][ucr-suite]
- [ ] SIMD optimized implementation
- [ ] Z-normalization
- [x] distance matrix calculations between N sequences
- [x] parallelized distance matrix calculations
- [ ] early stopping using `LB_Kim` (semi-implemented)
- [ ] early stopping using `LB_Keogh` (semi-implemented)
- [x] early stopping using the Euclidean upper bound

Pull requests for missing features would be very welcome.

# Example

```
use augurs_dtw::Dtw;
let a = &[0.0, 1.0, 2.0];
let b = &[3.0, 4.0, 5.0];
let dist = Dtw::euclidean().distance(a, b);
assert_eq!(dist, 5.0990195135927845);
```

[ucr-suite]: https://www.cs.ucr.edu/~eamonn/UCRsuite.html
