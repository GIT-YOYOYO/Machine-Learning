import numpy as np

from src.features.distance import (
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
)


def test_distances_from_3_4_5_triangle():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    q = np.array([0.0, 0.0])

    np.testing.assert_allclose(euclidean_distance(X, q), [0.0, 5.0])
    np.testing.assert_allclose(manhattan_distance(X, q), [0.0, 7.0])
    np.testing.assert_allclose(chebyshev_distance(X, q), [0.0, 4.0])


def test_minkowski_p2_matches_euclidean():
    X = np.array([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4]])
    q = np.array([6.0, 3.3, 4.5, 1.5])

    np.testing.assert_allclose(
        minkowski_distance(X, q, p=2), euclidean_distance(X, q)
    )


def test_distance_output_shape_is_number_of_samples():
    X = np.arange(12, dtype=float).reshape(3, 4)
    q = np.zeros(4)

    for metric in (euclidean_distance, manhattan_distance, chebyshev_distance):
        assert metric(X, q).shape == (3,)