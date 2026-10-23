"""数据集加载。"""

import numpy as np
from sklearn.datasets import load_iris


def load_iris_data():
    """加载鸢尾花数据集。

    返回
    ----
    X : ndarray, shape (150, 4)
    y : ndarray, shape (150,)
    feature_names : list[str]
    target_names : list[str]
    """
    data = load_iris()
    return (
        data.data,
        data.target,
        [str(name) for name in data.feature_names],
        [str(name) for name in data.target_names],
    )


def load_apple_data():
    """苹果二分类玩具数据。

    重量数值比甜度大一个数量级，未标准化时会主导距离；
    这批样本里重量的分布恰好与标签相反，甜度才是决定标签的特征。
    因此未标准化时会把 (200g, 15°Brix) 误判为普通苹果，
    标准化后才能按甜度找到真正的邻居。

    返回
    ----
    X : ndarray, shape (6, 2)
    y : ndarray, shape (6,)
        0 = 普通苹果，1 = 糖心苹果
    feature_names : list[str]
    """
    X = np.array([
        [200.0, 9.0],
        [210.0, 10.0],
        [190.0, 8.5],
        [120.0, 15.0],
        [280.0, 14.0],
        [300.0, 15.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y, ["重量 (g)", "甜度 (°Brix)"]