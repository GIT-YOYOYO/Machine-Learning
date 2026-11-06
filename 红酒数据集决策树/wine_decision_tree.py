# -*- coding: utf-8 -*-
"""专业方向强化课程2 实验2：红酒数据集决策树分类

运行方式：python wine_decision_tree.py
输出文件：decision_tree.png（决策树可视化）、confusion_matrix.png（混淆矩阵）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 使用非交互后端，便于直接保存图片
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# matplotlib 中文显示设置（Windows 自带字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "wine.csv"
RANDOM_STATE = 42


def main():
    # 1. 读取数据，确定特征与预测目标
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="target")
    y = df["target"]

    print("数据规模：", df.shape)
    print("类别分布：")
    print(y.value_counts().sort_index().to_string())
    print("缺失值数量：", int(df.isna().sum().sum()))
    print("重复行数量：", int(df.duplicated().sum()))

    # 2. 固定随机种子划分训练集与测试集（stratify 保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n训练集：{X_train.shape[0]} 条，测试集：{X_test.shape[0]} 条")

    # 3. 用 Pipeline 构建并训练决策树
    # 决策树基于特征阈值分裂，对特征量纲不敏感，因此无需标准化
    model = Pipeline(
        steps=[
            (
                "classifier",
                DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
            )
        ]
    )
    model.fit(X_train, y_train)

    # 4. 在测试集上评估模型
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率：{acc:.4f}")
    print("\n分类报告：")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["class_0", "class_1", "class_2"],
            digits=3,
        )
    )
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵：")
    print(cm)

    tree = model.named_steps["classifier"]
    print(f"\n决策树深度：{tree.get_depth()}，叶节点数：{tree.get_n_leaves()}")
    importance = pd.Series(tree.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print("\n特征重要性（降序）：")
    print(importance[importance > 0].round(4).to_string())
    print("\n决策规则：")
    print(export_text(tree, feature_names=list(X.columns)))

    # 5. 决策树可视化
    # 将最长的特征名换行显示，避免节点文字相互遮挡
    display_names = [
        name.replace("od280/od315_of_diluted_wines", "od280/od315_of\ndiluted_wines")
        for name in X.columns
    ]
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_tree(
        tree,
        feature_names=display_names,
        class_names=["class_0", "class_1", "class_2"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=11,
        ax=ax,
    )
    fig.savefig(BASE_DIR / "decision_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 混淆矩阵可视化
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), ["class_0", "class_1", "class_2"])
    ax.set_yticks(range(3), ["class_0", "class_1", "class_2"])
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("测试集混淆矩阵")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n已生成 decision_tree.png 和 confusion_matrix.png")


if __name__ == "__main__":
    main()