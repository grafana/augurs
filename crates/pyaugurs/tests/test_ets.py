import numpy as np
import pytest

from augurs import AutoETS


class TestAutoETS:
    """Test suite for AutoETS exponential smoothing models."""

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

    def test_instantiation(self):
        """Test that AutoETS can be instantiated with valid parameters."""
        model = AutoETS(5, "ZZZ")
        assert model is not None
        assert isinstance(model, AutoETS)

    def test_instantiation_with_different_specs(self):
        """Test instantiation with different model specifications."""
        specs = ["ZZN", "ZZZ", "ANN", "AAN", "AAA"]
        for spec in specs:
            model = AutoETS(4, spec)
            assert model is not None
            assert isinstance(model, AutoETS)

    def test_fit_with_numpy_array(self, sample_data):
        """Test fitting the model with a numpy array."""
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)
        # If no exception is raised, the test passes

    def test_fit_with_list(self, sample_data):
        """Test fitting the model with a Python list converted to array."""
        model = AutoETS(3, "ZZZ")
        # Lists need to be converted to numpy arrays first
        model.fit(np.array(sample_data.tolist()))
        # If no exception is raised, the test passes

    def test_fit_with_float64_array(self, seasonal_data):
        """Test fitting with explicit float64 array."""
        model = AutoETS(5, "ZZZ")
        model.fit(np.float64(seasonal_data))

    def test_predict_returns_forecast(self, sample_data):
        """Test that predict returns a Forecast object."""
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)
        forecast = model.predict(10, level=None)

        assert forecast is not None
        assert hasattr(forecast, "point")
        assert callable(forecast.point)

    def test_predict_returns_numpy_array(self, sample_data):
        """Test that predict returns numpy arrays."""
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)
        forecast = model.predict(10, level=None)

        point = forecast.point()
        assert isinstance(point, np.ndarray)
        assert len(point) == 10

    def test_predict_with_horizon(self, sample_data):
        """Test prediction with different horizons."""
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)

        for horizon in [1, 5, 10, 20]:
            forecast = model.predict(horizon, level=None)
            point = forecast.point()
            assert len(point) == horizon

    def test_predict_without_intervals(self, sample_data):
        """Test prediction without confidence intervals."""
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)
        forecast = model.predict(10, level=None)

        assert forecast.lower() is None
        assert forecast.upper() is None

    def test_predict_with_intervals(self, sample_data):
        """Test prediction with confidence intervals."""
        model = AutoETS(3, "ZZZ")
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
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)

        for level in [0.8, 0.9, 0.95, 0.99]:
            forecast = model.predict(10, level=level)
            assert forecast.lower() is not None
            assert forecast.upper() is not None
            assert len(forecast.lower()) == 10
            assert len(forecast.upper()) == 10

    def test_predict_in_sample(self, sample_data):
        """Test in-sample prediction."""
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)
        forecast = model.predict_in_sample(level=None)

        point = forecast.point()
        assert isinstance(point, np.ndarray)
        # In-sample predictions should match the length of input data
        assert len(point) == len(sample_data)

    def test_predict_in_sample_with_intervals(self, sample_data):
        """Test in-sample prediction with confidence intervals."""
        model = AutoETS(3, "ZZZ")
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

    def test_seasonal_model(self, seasonal_data):
        """Test with seasonal data and appropriate season length."""
        model = AutoETS(5, "ZZZ")
        model.fit(seasonal_data)
        forecast = model.predict(10, level=0.95)

        assert len(forecast.point()) == 10
        assert forecast.lower() is not None
        assert forecast.upper() is not None

    def test_multiple_fits(self, sample_data, seasonal_data):
        """Test that the same model can be fit multiple times."""
        model = AutoETS(3, "ZZZ")

        # First fit
        model.fit(sample_data)
        forecast1 = model.predict(5, level=None)

        # Second fit with different data
        model.fit(seasonal_data)
        forecast2 = model.predict(5, level=None)

        # Forecasts should be different
        assert not np.allclose(forecast1.point(), forecast2.point())

    def test_invalid_horizon(self, sample_data):
        """Test that invalid horizon raises an error."""
        model = AutoETS(3, "ZZZ")
        model.fit(sample_data)

        with pytest.raises(Exception):
            model.predict(0, level=None)

        with pytest.raises(Exception):
            model.predict(-1, level=None)

    def test_predict_before_fit(self):
        """Test that predicting before fitting raises an error."""
        model = AutoETS(3, "ZZZ")

        with pytest.raises(Exception):
            model.predict(10, level=None)

    def test_with_real_world_data(self):
        """Test with more realistic data patterns."""
        # Trend + seasonality
        t = np.arange(50)
        y = 10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(50) * 0.5

        model = AutoETS(12, "ZZZ")
        model.fit(y)
        forecast = model.predict(12, level=0.95)

        assert len(forecast.point()) == 12
        assert forecast.lower() is not None
        assert forecast.upper() is not None
