import numpy as np
import pytest

from augurs import MSTL, Forecast


class TestMSTL:
    """Test suite for MSTL (Multiple Seasonal Trend Decomposition with LOESS) models."""

    @pytest.fixture
    def sample_data(self):
        """Sample time series data for testing."""
        return np.array(
            [
                1.5,
                3.0,
                2.5,
                4.2,
                2.7,
                1.9,
                1.0,
                1.2,
                0.8,
                1.5,
                3.0,
                2.5,
                4.2,
                2.7,
                1.9,
                1.0,
                1.2,
                0.8,
            ]
        )

    @pytest.fixture
    def seasonal_data(self):
        """Seasonal time series data with period of 5."""
        return np.array([float(i % 5) for i in range(20)])

    @pytest.fixture
    def multi_seasonal_data(self):
        """Data with multiple seasonal patterns."""
        t = np.arange(100)
        # Daily (period 7) and weekly (period 4) patterns
        y = 10 + 2 * np.sin(2 * np.pi * t / 7) + 3 * np.sin(2 * np.pi * t / 4)
        return y

    def test_instantiation_with_ets(self):
        """Test that MSTL can be instantiated with ETS trend forecaster."""
        model = MSTL.ets([10])
        assert model is not None
        assert isinstance(model, MSTL)

    def test_instantiation_with_single_period(self):
        """Test instantiation with a single period."""
        model = MSTL.ets([7])
        assert model is not None

    def test_instantiation_with_multiple_periods(self):
        """Test instantiation with multiple periods."""
        model = MSTL.ets([3, 7])
        assert model is not None

    def test_instantiation_with_list(self):
        """Test instantiation with a Python list of periods."""
        model = MSTL.ets([3, 4])
        assert isinstance(model, MSTL)

    def test_instantiation_with_numpy_array(self):
        """Test instantiation with a numpy array of periods."""
        model = MSTL.ets(np.array([3, 4], dtype=np.uint32))
        assert isinstance(model, MSTL)

    def test_fit_with_numpy_array(self, sample_data):
        """Test fitting the model with a numpy array."""
        model = MSTL.ets([3])
        model.fit(sample_data)
        # If no exception is raised, the test passes

    def test_fit_with_list(self, sample_data):
        """Test fitting the model with a Python list converted to array."""
        model = MSTL.ets([3])
        # Lists need to be converted to numpy arrays first
        model.fit(np.array(sample_data.tolist()))
        # If no exception is raised, the test passes

    def test_fit_with_float64_array(self, seasonal_data):
        """Test fitting with explicit float64 array."""
        model = MSTL.ets([5])
        model.fit(np.float64(seasonal_data))

    def test_fit_with_multiple_periods(self, multi_seasonal_data):
        """Test fitting with multiple seasonal periods."""
        model = MSTL.ets([4, 7])
        model.fit(multi_seasonal_data)

    def test_predict_returns_forecast(self, sample_data):
        """Test that predict returns a Forecast object."""
        model = MSTL.ets([3])
        model.fit(sample_data)
        forecast = model.predict(10)

        assert forecast is not None
        assert hasattr(forecast, "point")
        assert callable(forecast.point)

    def test_predict_returns_numpy_array(self, sample_data):
        """Test that predict returns numpy arrays."""
        model = MSTL.ets([3])
        model.fit(sample_data)
        forecast = model.predict(10)

        point = forecast.point()
        assert isinstance(point, np.ndarray)
        assert len(point) == 10

    def test_predict_with_different_horizons(self, sample_data):
        """Test prediction with different horizons."""
        model = MSTL.ets([3])
        model.fit(sample_data)

        for horizon in [1, 5, 10, 20]:
            forecast = model.predict(horizon)
            point = forecast.point()
            assert len(point) == horizon

    def test_predict_without_intervals(self, sample_data):
        """Test prediction without confidence intervals."""
        model = MSTL.ets([3])
        model.fit(sample_data)
        forecast = model.predict(10)

        assert forecast.lower() is None
        assert forecast.upper() is None

    def test_predict_with_intervals(self, sample_data):
        """Test prediction with confidence intervals."""
        model = MSTL.ets([3])
        model.fit(sample_data)
        level = 0.95
        forecast = model.predict(10, level=level)

        point = forecast.point()
        lower = forecast.lower()
        upper = forecast.upper()

        assert isinstance(point, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert len(point) == 10
        assert len(lower) == 10
        assert len(upper) == 10

        # Lower bounds should be less than or equal to point forecasts
        # Upper bounds should be greater than or equal to point forecasts
        assert np.all(lower <= point)
        assert np.all(upper >= point)

    def test_predict_with_different_confidence_levels(self, sample_data):
        """Test prediction with different confidence levels."""
        model = MSTL.ets([3])
        model.fit(sample_data)

        for level in [0.8, 0.9, 0.95, 0.99]:
            forecast = model.predict(10, level=level)
            assert forecast.lower() is not None
            assert forecast.upper() is not None
            assert len(forecast.lower()) == 10
            assert len(forecast.upper()) == 10

    def test_predict_in_sample(self, sample_data):
        """Test in-sample prediction."""
        model = MSTL.ets([3])
        model.fit(sample_data)
        forecast = model.predict_in_sample(level=None)

        point = forecast.point()
        assert isinstance(point, np.ndarray)
        # In-sample predictions should match the length of input data
        assert len(point) == len(sample_data)

    def test_predict_in_sample_without_intervals(self, seasonal_data):
        """Test in-sample prediction without intervals."""
        model = MSTL.ets([5])
        model.fit(seasonal_data)
        forecast = model.predict_in_sample(level=None)

        point = forecast.point()
        assert isinstance(point, np.ndarray)
        assert len(point) == len(seasonal_data)
        assert forecast.lower() is None
        assert forecast.upper() is None

    def test_predict_in_sample_with_intervals(self, sample_data):
        """Test in-sample prediction with confidence intervals."""
        model = MSTL.ets([3])
        model.fit(sample_data)
        level = 0.95
        forecast = model.predict_in_sample(level=level)

        point = forecast.point()
        lower = forecast.lower()
        upper = forecast.upper()

        assert isinstance(point, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert len(point) == len(sample_data)
        assert len(lower) == len(sample_data)
        assert len(upper) == len(sample_data)

    def test_multiple_seasonal_periods(self, multi_seasonal_data):
        """Test with multiple seasonal periods."""
        model = MSTL.ets([4, 7])
        model.fit(multi_seasonal_data)

        forecast = model.predict(14, level=0.95)
        assert len(forecast.point()) == 14
        assert forecast.lower() is not None
        assert forecast.upper() is not None

    def test_multiple_fits(self, sample_data, seasonal_data):
        """Test that the same model can be fit multiple times."""
        model = MSTL.ets([3])

        # First fit
        model.fit(sample_data)
        forecast1 = model.predict(5)

        # Second fit with different data
        model.fit(seasonal_data)
        forecast2 = model.predict(5)

        # Forecasts should be different
        assert not np.allclose(forecast1.point(), forecast2.point())

    def test_predict_before_fit(self):
        """Test that predicting before fitting raises an error."""
        model = MSTL.ets([3])

        with pytest.raises(Exception):
            model.predict(10)

    def test_predict_in_sample_before_fit(self):
        """Test that in-sample prediction before fitting raises an error."""
        model = MSTL.ets([3])

        with pytest.raises(Exception):
            model.predict_in_sample(level=None)

    def test_invalid_horizon(self, sample_data):
        """Test that invalid horizon raises an error."""
        model = MSTL.ets([3])
        model.fit(sample_data)

        with pytest.raises(Exception):
            model.predict(0, level=None)

        with pytest.raises(Exception):
            model.predict(-1, level=None)

    def test_with_real_world_pattern(self):
        """Test with more realistic data patterns."""
        # Create data with trend and multiple seasonal patterns
        t = np.arange(100)
        trend = 0.5 * t
        seasonal1 = 5 * np.sin(2 * np.pi * t / 7)  # Weekly
        seasonal2 = 3 * np.sin(2 * np.pi * t / 4)  # 4-day
        noise = np.random.randn(100) * 0.5
        y = 10 + trend + seasonal1 + seasonal2 + noise

        model = MSTL.ets([4, 7])
        model.fit(y)

        forecast = model.predict(14, level=0.95)
        assert len(forecast.point()) == 14
        assert forecast.lower() is not None
        assert forecast.upper() is not None

    def test_readme_example(self):
        """Test the example from the README."""
        y = np.array([1.5, 3.0, 2.5, 4.2, 2.7, 1.9, 1.0, 1.2, 0.8])
        periods = [3, 4]

        # Use an AutoETS trend forecaster
        model = MSTL.ets(periods)
        model.fit(y)

        out_of_sample = model.predict(10, level=0.95)
        assert len(out_of_sample.point()) == 10
        assert out_of_sample.lower() is not None

        in_sample = model.predict_in_sample(level=0.95)
        assert len(in_sample.point()) == len(y)
