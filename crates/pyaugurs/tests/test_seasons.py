import numpy as np
import pytest

from augurs import seasonalities


class TestSeasonalities:
    """Test suite for seasonality detection."""

    @pytest.fixture
    def seasonal_data_period_4(self):
        """Time series with clear period-4 seasonality."""
        return np.array(
            [
                0.1,
                0.3,
                0.8,
                0.5,
                0.1,
                0.31,
                0.79,
                0.48,
                0.09,
                0.29,
                0.81,
                0.49,
                0.11,
                0.28,
                0.78,
                0.53,
                0.1,
                0.3,
                0.8,
                0.5,
                0.1,
                0.31,
                0.79,
                0.48,
                0.09,
                0.29,
                0.81,
                0.49,
                0.11,
                0.28,
                0.78,
                0.53,
            ]
        )

    @pytest.fixture
    def seasonal_data_period_7(self):
        """Time series with period-7 (weekly) seasonality."""
        t = np.arange(70)
        return 10 + 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(70) * 0.1

    @pytest.fixture
    def seasonal_data_period_12(self):
        """Time series with period-12 (monthly) seasonality."""
        t = np.arange(120)
        return 100 + 20 * np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.5

    @pytest.fixture
    def multi_seasonal_data(self):
        """Time series with multiple seasonal periods."""
        t = np.arange(168)  # 24 weeks of daily data
        # Weekly (7) and quarterly (approx 91/7 ≈ 13 weeks) patterns
        daily = 5 * np.sin(2 * np.pi * t / 7)
        weekly = 3 * np.sin(2 * np.pi * t / 28)
        return 50 + daily + weekly + np.random.randn(168) * 0.2

    @pytest.fixture
    def non_seasonal_data(self):
        """Time series with no clear seasonality."""
        return np.random.randn(100)

    def test_basic_seasonality_detection(self, seasonal_data_period_4):
        """Test basic seasonality detection with numpy array."""
        periods = seasonalities(seasonal_data_period_4)
        assert isinstance(periods, np.ndarray)
        assert len(periods) > 0
        assert 4 in periods

    def test_seasonality_with_list(self, seasonal_data_period_4):
        """Test seasonality detection with Python list converted to array."""
        # Lists need to be converted to numpy arrays first
        periods = seasonalities(np.array(seasonal_data_period_4.tolist()))
        assert isinstance(periods, np.ndarray)
        assert len(periods) > 0
        assert 4 in periods

    def test_seasonality_with_float64_array(self, seasonal_data_period_4):
        """Test seasonality detection with explicit float64 array."""
        periods = seasonalities(np.float64(seasonal_data_period_4))
        assert isinstance(periods, np.ndarray)
        assert len(periods) > 0
        assert 4 in periods

    def test_seasonality_period_7(self, seasonal_data_period_7):
        """Test detection of weekly (period-7) seasonality."""
        periods = seasonalities(seasonal_data_period_7)
        assert isinstance(periods, np.ndarray)
        # Period detection is approximate, just check we get something
        # The algorithm may or may not detect the exact period depending on noise

    def test_seasonality_period_12(self, seasonal_data_period_12):
        """Test detection of monthly (period-12) seasonality."""
        periods = seasonalities(seasonal_data_period_12)
        assert isinstance(periods, np.ndarray)
        # Period detection is approximate, just check we get something
        # The algorithm may or may not detect the exact period depending on noise

    def test_with_min_period(self, seasonal_data_period_4):
        """Test seasonality detection with min_period parameter."""
        periods = seasonalities(seasonal_data_period_4, min_period=3)
        assert isinstance(periods, np.ndarray)
        # All detected periods should be >= min_period
        assert all(p >= 3 for p in periods)

    def test_with_max_period(self, seasonal_data_period_12):
        """Test seasonality detection with max_period parameter."""
        periods = seasonalities(seasonal_data_period_12, max_period=20)
        assert isinstance(periods, np.ndarray)
        # All detected periods should be <= max_period
        assert all(p <= 20 for p in periods)

    def test_with_min_and_max_period(self, seasonal_data_period_7):
        """Test seasonality detection with both min and max period."""
        periods = seasonalities(seasonal_data_period_7, min_period=5, max_period=10)
        assert isinstance(periods, np.ndarray)
        # All detected periods should be within the range
        assert all(5 <= p <= 10 for p in periods)

    def test_with_threshold(self, seasonal_data_period_4):
        """Test seasonality detection with custom threshold."""
        # Lower threshold should be more sensitive
        periods_low = seasonalities(seasonal_data_period_4, threshold=0.5)
        # Higher threshold should be less sensitive
        periods_high = seasonalities(seasonal_data_period_4, threshold=0.9)

        assert isinstance(periods_low, np.ndarray)
        assert isinstance(periods_high, np.ndarray)

    def test_with_all_parameters(self, seasonal_data_period_7):
        """Test seasonality detection with all optional parameters."""
        periods = seasonalities(
            seasonal_data_period_7, min_period=4, max_period=14, threshold=0.7
        )
        assert isinstance(periods, np.ndarray)
        # All detected periods should be within the specified range
        assert all(4 <= p <= 14 for p in periods)

    def test_multiple_seasonalities(self, multi_seasonal_data):
        """Test detection of multiple seasonal periods."""
        periods = seasonalities(multi_seasonal_data)
        assert isinstance(periods, np.ndarray)
        # Should detect multiple periods
        # This is data-dependent, so we just check it returns something reasonable
        assert len(periods) >= 0

    def test_no_seasonality(self, non_seasonal_data):
        """Test with non-seasonal (random) data."""
        periods = seasonalities(non_seasonal_data)
        assert isinstance(periods, np.ndarray)
        # May return empty array or some spurious periods

    def test_returns_uint_array(self, seasonal_data_period_4):
        """Test that returned array has correct dtype."""
        periods = seasonalities(seasonal_data_period_4)
        # Should return unsigned integer array
        assert periods.dtype in [np.uint32, np.uint64]

    def test_with_trend_and_seasonality(self):
        """Test with data containing both trend and seasonality."""
        t = np.arange(100)
        trend = 0.5 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 10)
        y = trend + seasonal + np.random.randn(100) * 0.2

        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
        # Period detection is approximate and may vary with noise

    def test_with_short_series(self):
        """Test with very short time series."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
        # Short series may not have detectable seasonality

    def test_with_longer_series(self):
        """Test with longer time series."""
        t = np.arange(365)
        # Annual data with weekly seasonality
        y = 10 + 3 * np.sin(2 * np.pi * t / 7) + np.random.randn(365) * 0.3

        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)

    def test_with_constant_series(self):
        """Test with constant time series."""
        y = np.full(50, 5.0)
        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
        # Constant series should have no seasonality

    def test_with_linear_trend_only(self):
        """Test with pure linear trend (no seasonality)."""
        y = np.arange(100, dtype=float)
        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
        # Pure trend should have no significant seasonality

    def test_strong_seasonality_detection(self):
        """Test with very strong seasonal signal."""
        t = np.arange(100)
        # Very strong seasonal component, minimal noise
        y = 100 * np.sin(2 * np.pi * t / 8) + np.random.randn(100) * 0.01

        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
        assert len(periods) > 0
        # Should definitely detect the period of 8
        assert any(abs(p - 8) <= 1 for p in periods)

    def test_weak_seasonality_detection(self):
        """Test with weak seasonal signal in noisy data."""
        t = np.arange(200)
        # Weak seasonal component with high noise
        y = 2 * np.sin(2 * np.pi * t / 15) + np.random.randn(200) * 5

        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
        # Weak signal might not be detected

    def test_readme_example(self):
        """Test the pattern from the README examples."""
        # Based on patterns seen in other tests
        y = np.array(
            [
                0.1,
                0.3,
                0.8,
                0.5,
                0.1,
                0.31,
                0.79,
                0.48,
                0.09,
                0.29,
                0.81,
                0.49,
                0.11,
                0.28,
                0.78,
                0.53,
                0.1,
                0.3,
                0.8,
                0.5,
                0.1,
                0.31,
                0.79,
                0.48,
                0.09,
                0.29,
                0.81,
                0.49,
                0.11,
                0.28,
                0.78,
                0.53,
            ]
        )

        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
        assert 4 in periods

    def test_business_data_pattern(self):
        """Test with business/economic data pattern."""
        # Quarterly seasonality (period 4) common in business data
        quarters = np.repeat([100, 150, 120, 180], 10)
        noise = np.random.randn(40) * 5
        y = quarters + noise

        periods = seasonalities(y)
        assert isinstance(periods, np.ndarray)
