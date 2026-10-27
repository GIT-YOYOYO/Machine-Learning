"""KNN 超参数的网格搜索。"""

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11, 13],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}


def grid_search_knn(X, y, cv=5):
    """在给定数据上搜索 KNN 超参数，返回拟合好的 GridSearchCV 对象。"""
    search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=KNN_PARAM_GRID,
        cv=cv,
        scoring="accuracy",
    )
    search.fit(X, y)
    return search