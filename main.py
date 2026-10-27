"""KNN 最小链路演示：距离度量、手写实现与 sklearn 对照。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loaders import load_apple_data, load_iris_data
from src.evaluation.leakage import compare_feature_selection_leakage
from src.evaluation.metrics import accuracy_score as custom_accuracy_score
from src.evaluation.metrics import confusion_matrix as custom_confusion_matrix
from src.models.knn import KNNClassifier
from src.models.search import grid_search_knn
from src.visualization.plots import plot_k_selection, plot_knn_boundaries

FIGURES_DIR = Path(__file__).resolve().parent / "reports" / "figures"


def cross_validate_accuracy(make_model, X, y, cv):
    """用给定划分循环训练/预测，返回各折准确率的平均值。"""
    scores = [
        np.mean(make_model().fit(X[train], y[train]).predict(X[test]) == y[test])
        for train, test in cv.split(X, y)
    ]
    return float(np.mean(scores))


def demo_iris():
    X, y, feature_names, target_names = load_iris_data()
    print(f"鸢尾花：{X.shape[0]} 个样本，{X.shape[1]} 个特征，{len(target_names)} 个类别")
    print(f"特征：{feature_names}\n")

    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    print("10 折交叉验证准确率（k=5）")
    print(f"{'距离':<10}{'手写 KNN':>10}{'sklearn':>10}")
    for metric in ("euclidean", "manhattan", "chebyshev"):
        ours = cross_validate_accuracy(lambda: KNNClassifier(5, metric=metric), X, y, cv)
        ref = cross_validate_accuracy(
            lambda: KNeighborsClassifier(5, metric=metric), X, y, cv
        )
        print(f"{metric:<10}{ours:>10.3f}{ref:>10.3f}")


def demo_tie():
    X = np.array([[3, 4], [7, 4], [3, 7], [8, 3], [4, 2], [6, 2], [4, 8], [8, 1]])
    y = np.array(["朱"] * 4 + ["墨"] * 4)
    q = np.array([[4, 4]])

    print("\n平票演示：k=4，待测点 (4,4)，2 票对 2 票")
    for weights in ("uniform", "distance"):
        ours = KNNClassifier(4, weights=weights).fit(X, y).predict(q)[0]
        ref = KNeighborsClassifier(4, weights=weights).fit(X, y).predict(q)[0]
        print(f"weights={weights:<9} 手写：{ours}  sklearn：{ref}")


def demo_student():
    X = np.array([
        [90, 95], [85, 88], [78, 90], [82, 76],
        [40, 30], [50, 45], [35, 60], [55, 35],
    ])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    student = np.array([[70, 75]])
    names = ["挂科", "通过"]

    print("\n学生通过预测：k=8，4 个通过 / 4 个挂科")
    for weights in ("uniform", "distance"):
        ours = KNNClassifier(8, weights=weights).fit(X, y).predict(student)[0]
        ref = KNeighborsClassifier(8, weights=weights).fit(X, y).predict(student)[0]
        print(f"weights={weights:<9} 手写：{names[ours]}  sklearn：{names[ref]}")


def demo_split_variance():
    X, y, _, _ = load_iris_data()
    # 与图 5-1 的二维决策边界保持同一口径
    X = X[:, :2]
    print("\n单次划分的波动：萼片两个特征，test_size=0.3，用测试集选 k")
    for seed in (0, 1, 7, 42, 2026):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed
        )
        scores = [
            KNeighborsClassifier(k).fit(X_train, y_train).score(X_test, y_test)
            for k in range(1, 81)
        ]
        best_k = int(np.argmax(scores)) + 1
        print(f"seed={seed:<5} 最佳 k={best_k:<3} 测试准确率={max(scores):.3f}")


def demo_confusion_matrix():
    X, y, _, target_names = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    y_pred = KNeighborsClassifier(7).fit(X_train, y_train).predict(X_test)

    print("\n混淆矩阵：iris，k=7，test_size=0.2，stratify=y")
    print(f"类别顺序：{target_names}")
    print(
        f"手写准确率：{custom_accuracy_score(y_test, y_pred):.3f}，"
        f"sklearn 准确率：{accuracy_score(y_test, y_pred):.3f}"
    )
    print("手写混淆矩阵：")
    print(custom_confusion_matrix(y_test, y_pred))
    print("sklearn 混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))


def demo_apple_scale():
    X, y, _ = load_apple_data()
    query = np.array([[200.0, 15.0]])

    print("\n苹果案例：重量会误导邻居，甜度才决定标签")
    print("待测苹果：重量 200 g，甜度 15 °Brix（真实类别：糖心）")
    for name, use_scaler in (("未标准化", False), ("标准化", True)):
        if use_scaler:
            scaler = StandardScaler().fit(X)
            X_used = scaler.transform(X)
            query_used = scaler.transform(query)
        else:
            X_used = X
            query_used = query

        model = KNeighborsClassifier(3).fit(X_used, y)
        neighbors = model.kneighbors(query_used, return_distance=False)[0]
        pred = int(model.predict(query_used)[0])
        print(
            f"{name:<5} 预测：{'糖心' if pred == 1 else '普通'}，"
            f"3 个邻居的真实标签：{y[neighbors].tolist()}"
        )


def demo_scaling():
    X, y, _, _ = load_iris_data()
    X_bad = X.copy()
    X_bad[:, 0] *= 1000

    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    raw = cross_validate_accuracy(lambda: KNNClassifier(5), X, y, cv)
    bad = cross_validate_accuracy(lambda: KNNClassifier(5), X_bad, y, cv)
    fixed = cross_validate_accuracy(
        lambda: make_pipeline(StandardScaler(), KNNClassifier(5)), X_bad, y, cv
    )

    print("\n标准化演示：人为把萼片长度放大 1000 倍（k=5）")
    print(f"原始数据           {raw:.3f}")
    print(f"放大后未标准化     {bad:.3f}")
    print(f"放大后 + 标准化    {fixed:.3f}")


def demo_leakage():
    leak, correct = compare_feature_selection_leakage()
    print("\n数据泄露：100 个样本、5000 个纯噪声特征")
    print(f"全量数据上 SelectKBest（泄露）：{leak:.3f}")
    print(f"Pipeline 内逐折 SelectKBest：{correct:.3f}")
    print("数据没有真实规律，正确分数应接近 0.5；泄露分数是假象。")


def demo_grid_search():
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    search = grid_search_knn(X_train, y_train, cv=cv)

    print("\n网格搜索：iris，搜索 k / weights / p")
    print(f"最优参数：{search.best_params_}")
    print(f"训练集 5 折交叉验证最高准确率：{search.best_score_:.3f}")
    print(f"测试集准确率：{search.best_estimator_.score(X_test, y_test):.3f}")
    print("测试集只用于最终评估，不参与选参。")


def demo_k_selection():
    X, y, _, _ = load_iris_data()
    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    ks = list(range(1, 31))
    scores = [cross_validate_accuracy(lambda k=k: KNNClassifier(k), X, y, cv) for k in ks]
    best_k = ks[int(np.argmax(scores))]

    print(f"\nk 选择：最佳 k = {best_k}，交叉验证准确率 {max(scores):.3f}")
    plot_k_selection(ks, scores, best_k, save_path=FIGURES_DIR / "k_selection.png")
    print(f"k 曲线已保存：{FIGURES_DIR / 'k_selection.png'}")


def demo_visualization():
    X, y, feature_names, target_names = load_iris_data()
    save_path = FIGURES_DIR / "knn_boundaries.png"
    fig = plot_knn_boundaries(
        X[:, :2],
        y,
        ks=(1, 5, 15),
        feature_names=(feature_names[0], feature_names[1]),
        class_names=target_names,
        save_path=save_path,
    )
    plt.close(fig)
    print(f"\n决策边界图已保存：{save_path}")


if __name__ == "__main__":
    demo_iris()
    demo_tie()
    demo_student()
    demo_split_variance()
    demo_k_selection()
    demo_confusion_matrix()
    demo_apple_scale()
    demo_scaling()
    demo_leakage()
    demo_grid_search()
    demo_visualization()