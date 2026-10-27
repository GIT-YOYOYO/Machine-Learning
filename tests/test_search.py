from sklearn.model_selection import StratifiedKFold

from src.data.loaders import load_iris_data
from src.models.search import KNN_PARAM_GRID, grid_search_knn


def test_grid_search_returns_best_params_from_grid():
    X, y, _, _ = load_iris_data()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)

    search = grid_search_knn(X, y, cv=cv)

    for name, value in search.best_params_.items():
        assert value in KNN_PARAM_GRID[name]
    assert 0.8 <= search.best_score_ <= 1.0