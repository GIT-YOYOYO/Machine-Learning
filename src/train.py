"""配置驱动的线性回归训练脚本。

用法：
    python src/train.py
    python src/train.py --config configs/linear.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "linear": lambda alpha: LinearRegression(),
    "ridge": lambda alpha: Ridge(alpha=alpha),
    "lasso": lambda alpha: Lasso(alpha=alpha),
}


def set_seed(seed: int) -> None:
    """固定随机种子，保证数据划分和训练结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def load_data(cfg: dict):
    """按配置读取数据，并按需构造 √微信 特征。"""
    df = pd.read_csv(cfg["data"]["path"])
    X = df[cfg["data"]["features"]].copy()
    if cfg["features"].get("add_sqrt_wechat", False):
        X["√微信"] = np.sqrt(X["微信"])
    y = df[cfg["data"]["target"]]
    return X, y


def evaluate(model, X, y) -> dict:
    """返回测试集上的 MAE、RMSE、R²。"""
    pred = model.predict(X)
    return {
        "MAE": round(mean_absolute_error(y, pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 3),
        "R2": round(r2_score(y, pred), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["split"]["random_state"])
    X, y = load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )

    # 基线：不管输入是什么，永远预测训练集均值
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    if model_name not in MODELS:
        raise ValueError(f"不支持的模型: {model_name!r}，可选 {list(MODELS)}")

    # 标准化和模型绑成流水线，避免数据泄露
    model = make_pipeline(
        StandardScaler(),
        MODELS[model_name](model_cfg.get("alpha", 1.0)),
    ).fit(X_train, y_train)

    result = {
        "baseline": evaluate(baseline, X_test, y_test),
        model_name: evaluate(model, X_test, y_test),
    }

    for name, metrics in result.items():
        print(
            f"{name:<9} MAE = {metrics['MAE']:.3f}  "
            f"RMSE = {metrics['RMSE']:.3f}  R² = {metrics['R2']:.3f}"
        )

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"config": cfg, "result": result}, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到 {report_path}")


if __name__ == "__main__":
    main()