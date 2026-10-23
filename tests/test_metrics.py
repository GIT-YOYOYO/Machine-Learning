import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from src.evaluation.metrics import accuracy_score, confusion_matrix


def test_confusion_matrix_matches_sklearn():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 2])

    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred),
        sklearn_confusion_matrix(y_true, y_pred),
    )


def test_confusion_matrix_layout_is_true_by_predicted():
    matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])

    assert matrix.tolist() == [[1, 1], [1, 1]]


def test_accuracy_score():
    assert accuracy_score([0, 1, 1], [0, 1, 0]) == 2 / 3