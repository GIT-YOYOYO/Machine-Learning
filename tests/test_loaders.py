from src.data.loaders import load_apple_data, load_iris_data


def test_load_iris_data_shapes():
    X, y, feature_names, target_names = load_iris_data()

    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(feature_names) == 4
    assert target_names == ["setosa", "versicolor", "virginica"]


def test_load_apple_data_shapes():
    X, y, feature_names = load_apple_data()

    assert X.shape == (6, 2)
    assert y.shape == (6,)
    assert set(y) == {0, 1}
    assert len(feature_names) == 2