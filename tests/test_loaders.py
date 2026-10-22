from src.data.loaders import load_iris_data


def test_load_iris_data_shapes():
    X, y, feature_names, target_names = load_iris_data()

    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(feature_names) == 4
    assert target_names == ["setosa", "versicolor", "virginica"]