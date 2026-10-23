import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.data.loaders import load_apple_data
from src.models.knn import KNNClassifier

X_TOY = np.array([[3, 4], [7, 4], [3, 7], [8, 3], [4, 2], [6, 2], [4, 8], [8, 1]])
Y_TOY = np.array(["朱"] * 4 + ["墨"] * 4)


def test_tie_is_broken_by_smallest_class_label():
    model = KNNClassifier(n_neighbors=4).fit(X_TOY, Y_TOY)

    assert list(model.classes_) == ["墨", "朱"]
    assert model.predict([[4, 4]])[0] == "墨"


def test_distance_weights_change_prediction_for_exact_match():
    uniform = KNNClassifier(n_neighbors=4).fit(X_TOY, Y_TOY)
    weighted = KNNClassifier(n_neighbors=4, weights="distance").fit(X_TOY, Y_TOY)

    # (3,4) 与训练样本 A 完全重合：uniform 仍平票判墨，distance 判朱
    assert uniform.predict([[3, 4]])[0] == "墨"
    assert weighted.predict([[3, 4]])[0] == "朱"


def test_student_example_k8():
    X = np.array([
        [90, 95], [85, 88], [78, 90], [82, 76],
        [40, 30], [50, 45], [35, 60], [55, 35],
    ])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    student = np.array([[70, 75]])

    assert KNNClassifier(n_neighbors=8).fit(X, y).predict(student)[0] == 0
    assert (
        KNNClassifier(n_neighbors=8, weights="distance").fit(X, y).predict(student)[0]
        == 1
    )


def test_predictions_match_sklearn_on_iris():
    X, y = load_iris(return_X_y=True)
    train = np.arange(0, len(X), 2)
    test = np.arange(1, len(X), 2)

    ours = KNNClassifier(n_neighbors=5).fit(X[train], y[train])
    ref = KNeighborsClassifier(n_neighbors=5).fit(X[train], y[train])

    np.testing.assert_array_equal(ours.predict(X[test]), ref.predict(X[test]))


def test_invalid_neighbor_count_raises():
    with pytest.raises(ValueError, match="不能超过训练样本数"):
        KNNClassifier(n_neighbors=20).fit(X_TOY, Y_TOY)


def test_scaling_fixes_apple_distance_dominance():
    X, y, _ = load_apple_data()
    query = np.array([[200.0, 15.0]])

    raw = KNNClassifier(n_neighbors=3).fit(X, y).predict(query)[0]
    scaler = StandardScaler().fit(X)
    scaled = (
        KNNClassifier(n_neighbors=3)
        .fit(scaler.transform(X), y)
        .predict(scaler.transform(query))[0]
    )

    # 未标准化时重量主导，三个邻居都是普通苹果
    assert raw == 0
    # 标准化后甜度主导，三个邻居都是糖心苹果
    assert scaled == 1