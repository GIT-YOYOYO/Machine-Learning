"""特征选择中的数据泄露演示。"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


def compare_feature_selection_leakage(
    n_samples=100,
    n_features=5000,
    n_selected=20,
    random_state=0,
    cv_splits=5,
):
    """对比特征选择的泄露写法与 Pipeline 写法。

    数据是纯高斯噪声、标签随机，不存在真实规律，因此正确做法应接近 0.5。

    返回
    ----
    leak_score : float
        先在全部数据上做 SelectKBest，再做交叉验证的准确率。
    pipeline_score : float
        把 SelectKBest 放进 Pipeline，每折只用训练部分选择特征的准确率。
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=0)

    # 错误写法：特征选择阶段已经看过全部 X 和 y，包括验证折的标签
    X_selected = SelectKBest(f_classif, k=n_selected).fit_transform(X, y)
    leak_score = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), X_selected, y, cv=cv
    ).mean()

    # 正确写法：每折内部只用训练部分 fit 特征选择
    pipe = make_pipeline(
        SelectKBest(f_classif, k=n_selected),
        KNeighborsClassifier(n_neighbors=5),
    )
    pipeline_score = cross_val_score(pipe, X, y, cv=cv).mean()

    return float(leak_score), float(pipeline_score)