"""数据集加载。"""

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
    return data.data, data.target, list(data.feature_names), list(data.target_names)