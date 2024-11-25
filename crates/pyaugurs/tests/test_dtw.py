import numpy as np
import pytest

from augurs import Dtw


class TestDtw:
    @pytest.mark.parametrize(
        "opts, input, expected",
        [
            ({}, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 5.0990195135926845),
            ({"window": 2}, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 5.0990195135926845),
            ({"distance_fn": "euclidean"}, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 5.0990195135926845),
            ({"distance_fn": "manhattan"}, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], 9),
        ],
    )
    def test_distance(self, opts, input, expected):
        d = Dtw(**opts)
        arrays = [np.array(x) for x in input]
        np.testing.assert_allclose(d.distance(*arrays), expected)

    @pytest.mark.parametrize(
        "opts, input, expected",
        [
            (
                {},
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [
                    [0.0, 5.0990195135926845, 10.392304845413264],
                    [5.0990195135926845, 0.0, 5.0990195135926845],
                    [10.392304845413264, 5.0990195135926845, 0.0],
                ]
            ),
            (
                {"distance_fn": "manhattan"},
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [
                    [0.0, 9.0, 18.0],
                    [9.0, 0.0, 9.0],
                    [18.0, 9.0, 0.0],
                ]
            ),
        ],
    )
    def test_distance_matrix(self, opts, input, expected):
        d = Dtw(**opts)
        arrays = [np.array(x) for x in input]
        np.testing.assert_allclose(d.distance_matrix(arrays).to_numpy(), expected)
