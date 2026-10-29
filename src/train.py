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

# 项目根目录：src/train.py 的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def resolve_from_project_root(path_value) -> Path:
    """相对路径按项目根目录解析，绝对路径原样返回。"""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """固定随机种子，保证数据划分和训练结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def load_data(cfg: dict):
    """按配置读取数据，并按需构造 √微信 特征。"""
    data_path = resolve_from_project_root(cfg["data"]["path"])
    df = pd.read_csv(data_path)
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
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="配置文件路径；相对路径找不到时按项目根目录解析",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        # 优先按当前工作目录解析；找不到再按项目根目录解析
        if not config_path.exists():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

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

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"config": cfg, "result": result}, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到 {report_path}")


if __name__ == "__main__":
    main()