"""KNN 使用的距离度量。"""

import numpy as np


def minkowski_distance(X, q, p=2):
    """计算 X 中每个样本到 q 的闵氏距离。

    参数
    ----
    X : array-like, shape (n_samples, n_features)
        训练样本；也接受单个样本 (n_features,)。
    q : array-like, shape (n_features,)
        待测样本。
    p : float
        p=1 为曼哈顿距离，p=2 为欧氏距离，p=np.inf 为切比雪夫距离。

    返回
    ----
    ndarray, shape (n_samples,)
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    q = np.asarray(q, dtype=float).reshape(1, -1)
    diff = np.abs(X - q)

    if np.isinf(p):
        return diff.max(axis=1)
    return (diff**p).sum(axis=1) ** (1.0 / p)


def euclidean_distance(X, q):
    """欧氏距离（p=2）。"""
    return minkowski_distance(X, q, p=2)


def manhattan_distance(X, q):
    """曼哈顿距离（p=1）。"""
    return minkowski_distance(X, q, p=1)


def chebyshev_distance(X, q):
    """切比雪夫距离（p=∞）。"""
    return minkowski_distance(X, q, p=np.inf)


def pairwise_minkowski(X, Y, p=2):
    """计算 X 中每个样本到 Y 中每个样本的闵氏距离。

    参数
    ----
    X : array-like, shape (n_samples_x, n_features)
    Y : array-like, shape (n_samples_y, n_features)

    返回
    ----
    ndarray, shape (n_samples_x, n_samples_y)
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    diff = np.abs(X[:, None, :] - Y[None, :, :])

    if np.isinf(p):
        return diff.max(axis=2)
    return (diff**p).sum(axis=2) ** (1.0 / p)