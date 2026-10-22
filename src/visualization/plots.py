"""KNN 相关的可视化。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.models.knn import KNNClassifier

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_knn_boundaries(
    X,
    y,
    ks=(1, 5, 15),
    feature_names=("特征 1", "特征 2"),
    class_names=None,
    save_path=None,
):
    """绘制不同 k 下 KNN 在二维特征平面上的决策边界。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in classes]

    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    cmap = ListedColormap(colors)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, k in zip(axes, ks):
        model = KNNClassifier(n_neighbors=k).fit(X, y)
        zz = model.predict(grid).reshape(xx.shape)
        zz_idx = np.searchsorted(classes, zz)

        ax.contourf(
            xx,
            yy,
            zz_idx,
            levels=np.arange(len(classes) + 1) - 0.5,
            cmap=cmap,
            alpha=0.25,
        )
        for i, label in enumerate(classes):
            mask = y == label
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                color=colors[i],
                edgecolor="black",
                linewidth=0.3,
                s=25,
                label=class_names[i],
            )
        ax.set_title(f"k = {k}")
        ax.set_xlabel(feature_names[0])

    axes[0].set_ylabel(feature_names[1])
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("KNN 决策边界")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_k_selection(ks, scores, best_k, save_path=None):
    """绘制 k 与交叉验证准确率的关系。"""
    ks = list(ks)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, scores, marker="o", markersize=3, linewidth=1.5, color="#4C72B0")
    ax.axvline(best_k, color="#C44E52", linestyle="--", linewidth=1)
    ax.scatter([best_k], [scores[ks.index(best_k)]], color="#C44E52", zorder=3)

    ax.set_xlabel("k")
    ax.set_ylabel("10 折交叉验证准确率")
    ax.set_title(f"KNN 的 k 选择（最佳 k = {best_k}）")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig