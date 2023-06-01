import abc
from typing import Sequence

import numpy as np
import numpy.typing as npt


class TrendModel(abc.ABC):
    def fit(self, y: npt.NDArray[np.float64]) -> None:
        """
        Fit the trend model to the given time series.

        :param y: the time series to fit the model to.
        """

    def predict(self, horizon: int, level: float | None) -> Forecast:
        """
        Predict the next `horizon` values, optionally including prediction intervals
        at the given level.

        If provided, `level` must be between 0 and 1.

        :param level: the level at which to compute prediction intervals, if any. Must be
                      between 0 and 1.
        :return: a `Forecast` instance containing the predictions.
        """

    def predict_in_sample(self, level: float | None) -> Forecast:
        """
        Predict the next `horizon` values, optionally including prediction intervals
        at the given level.

        If provided, `level` must be between 0 and 1.

        :param level: the level at which to compute prediction intervals, if any. Must be
                      between 0 and 1.
        :return: a `Forecast` instance containing the predictions.
        """


class Forecast:
    def __init__(
        self,
        point: npt.NDArray[np.float64],
        level: float | None = None,
        lower: npt.NDArray[np.float64] | None = None,
        upper: npt.NDArray[np.float64] | None = None,
    ) -> None: ...
    def point(self) -> npt.NDArray[np.float64]: ...
    def lower(self) -> npt.NDArray[np.float64] | None: ...
    def upper(self) -> npt.NDArray[np.float64] | None: ...


class PyTrendModel:
    def __init__(self, trend_model: TrendModel) -> None: ...

class MSTL:
    @classmethod
    def ets(cls, periods: Sequence[int]) -> 'MSTL': ...
    @classmethod
    def custom_trend(cls, periods: Sequence[int], trend_model: PyTrendModel) -> 'MSTL': ...
    def fit(self, y: npt.NDArray[np.float64]) -> None: ...
    def predict(self, horizon: int, level: float | None) -> Forecast: ...
    def predict_in_sample(self, level: float | None) -> Forecast: ...

class AutoETS:
    def __init__(self, season_length: int, spec: str) -> None: ...
    def fit(self, y: npt.NDArray[np.float64]) -> None: ...
    def predict(self, horizon: int, level: float | None) -> Forecast: ...
    def predict_in_sample(self, level: float | None) -> Forecast: ...
