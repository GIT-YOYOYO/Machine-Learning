"""K 近邻分类器的手写实现。"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.features.distance import pairwise_minkowski

_METRIC_P = {
    "euclidean": 2,
    "manhattan": 1,
    "chebyshev": np.inf,
}


class KNNClassifier(ClassifierMixin, BaseEstimator):
    """K 近邻分类器，接口与 sklearn 的 KNeighborsClassifier 对齐。

    参数
    ----
    n_neighbors : int
        邻居个数 k。
    metric : {"euclidean", "manhattan", "chebyshev"}
        距离度量。
    weights : {"uniform", "distance"}
        uniform 一人一票；distance 按 1/d 加权。
    """

    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X 应为二维数组 (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("X 与 y 的样本数不一致")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors 必须大于等于 1")
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors 不能超过训练样本数")
        if self.metric not in _METRIC_P:
            raise ValueError(f"不支持的 metric: {self.metric}")
        if self.weights not in ("uniform", "distance"):
            raise ValueError(f"不支持的 weights: {self.weights}")

        # np.unique 会排序，classes_ 的顺序与 sklearn 一致
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        self.X_ = X
        return self

    def _distances(self, X):
        return pairwise_minkowski(X, self.X_, p=_METRIC_P[self.metric])

    def _vote(self, d):
        nearest = np.argsort(d, kind="stable")[: self.n_neighbors]
        labels, distances = self.y_[nearest], d[nearest]

        if self.weights == "distance":
            # 与 sklearn 一致：存在完全重合的邻居时只由它们投票
            exact = distances == 0
            if exact.any():
                labels = labels[exact]
                w = np.ones(len(labels))
            else:
                w = 1.0 / distances
        else:
            w = None

        votes = np.bincount(labels, weights=w, minlength=len(self.classes_))
        return self.classes_[np.argmax(votes)]

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self._vote(d) for d in self._distances(X)])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))