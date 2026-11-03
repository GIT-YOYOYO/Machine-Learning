# -*- coding: utf-8 -*-
"""Plot ROC and Precision-Recall curves for the credit default model."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "信贷数据.csv"
OUT_PATH = PROJECT_ROOT / "notebooks" / "images" / "notes10_fig_roc_pr.png"

RANDOM_STATE = 42
NUMERIC_FEATURES = ["年龄", "月收入", "负债收入比", "额度使用率", "逾期次数", "工作年限"]
CATEGORICAL_FEATURES = ["学历", "住房", "职业"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["是否违约"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    model = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ]).fit(X_train, y_train)

    proba_base = baseline.predict_proba(X_test)[:, 1]
    proba_model = model.predict_proba(X_test)[:, 1]

    fpr_base, tpr_base, _ = roc_curve(y_test, proba_base)
    fpr_model, tpr_model, _ = roc_curve(y_test, proba_model)
    precision_model, recall_model, _ = precision_recall_curve(y_test, proba_model)

    baseline_auc = roc_auc_score(y_test, proba_base)
    model_auc = roc_auc_score(y_test, proba_model)
    baseline_ap = average_precision_score(y_test, proba_base)
    model_ap = average_precision_score(y_test, proba_model)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        fpr_base, tpr_base,
        color="#999999", linewidth=2,
        label=f"基线 AUC={baseline_auc:.3f}",
    )
    axes[0].plot(
        fpr_model, tpr_model,
        color="#C44E52", linewidth=2,
        label=f"逻辑回归 AUC={model_auc:.3f}",
    )
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("假正率 FPR")
    axes[0].set_ylabel("真正率 TPR / 召回率")
    axes[0].set_title("(a) ROC 曲线")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(
        baseline_ap,
        color="#999999", linestyle="--", linewidth=2,
        label=f"随机基线 AP={baseline_ap:.3f}",
    )
    axes[1].plot(
        recall_model, precision_model,
        color="#C44E52", linewidth=2,
        label=f"逻辑回归 AP={model_ap:.3f}",
    )
    axes[1].set_xlabel("召回率 Recall")
    axes[1].set_ylabel("精确率 Precision")
    axes[1].set_title("(b) Precision-Recall 曲线")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"基线 AUC={baseline_auc:.4f}, AP={baseline_ap:.4f}")
    print(f"逻辑回归 AUC={model_auc:.4f}, AP={model_ap:.4f}")
    print(f"图像已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()