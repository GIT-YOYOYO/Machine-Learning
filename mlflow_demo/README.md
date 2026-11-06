# MLflow 实验追踪演示

本项目对应《机器学习课堂笔记》第 17 章，演示如何用 MLflow 记录机器学习实验的参数、指标、图片、CSV 报告和模型，并通过父子 Run 组织批量调参实验。

## 目录说明

```text
mlflow_demo/
├── artifacts_out/          # 代码生成的本地图片、CSV 报告（自动创建）
├── mlflow.db               # MLflow 3.x SQLite 元数据库（自动创建）
├── mlruns/                 # MLflow 制品目录：模型、图片、CSV（自动创建）
├── knn_manual_log.py       # KNN 手动记录 4 个核心 API
├── lr_diabetes.py          # 线性回归：指标、图片和 CSV 报告
├── parent_child_run.py     # 父子 Run：批量搜索 KNN 的 K 值
├── query_best_run.py       # 代码查询最优 Run 并加载模型
├── autolog_logreg.py       # autolog 自动记录逻辑回归
├── requirements.txt        # 依赖清单
└── README.md
```

> MLflow 3.x 的文件型 Tracking Store 已进入维护模式，所以本项目使用 SQLite 保存实验与 Run 元数据，模型和图片等制品仍然写入 `mlruns/`。

## 环境安装

从主项目根目录进入本目录：

```bash
cd mlflow_demo
```

在本目录打开终端，安装依赖：

```bash
pip install -r requirements.txt
```

## 启动 MLflow 网页

必须在 `mlflow_demo` 目录启动，并指定 SQLite 元数据库：

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

也可以先设置环境变量再启动：

```bat
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui
```

浏览器访问：<http://127.0.0.1:5000>

## 按顺序运行演示

Windows 下先打开 UTF-8 模式，并指定 SQLite 元数据库：

```bat
set PYTHONUTF8=1
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

PowerShell 使用：

```powershell
$env:PYTHONUTF8="1"
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

然后执行：

```bash
python knn_manual_log.py
python lr_diabetes.py
python parent_child_run.py
python query_best_run.py
python autolog_logreg.py
```

每次运行会向实验 `KNN-LR实验追踪演示` 写入一条 Run。刷新网页后可以查看：

- `Parameters`：超参数；
- `Metrics`：准确率、MSE、R² 等；
- `Artifacts`：混淆矩阵图片、预测对比图、CSV 报告和模型；
- 父 Run 下会折叠显示 K=1/3/5/7 四个子 Run；
- `query_best_run.py` 会按 `accuracy` 降序取最优 Run，并用 `runs:/<run_id>/knn_model` 加载模型。

## 四个手动记录 API

```python
mlflow.log_param("n_neighbors", k)          # 记录超参数
mlflow.log_metric("accuracy", acc)          # 记录评估指标
mlflow.log_artifact("artifacts_out/a.png")  # 记录图片、CSV 等文件
mlflow.sklearn.log_model(knn, name="knn_model")  # 记录模型
```

`autolog_logreg.py` 则用一行 `mlflow.autolog()` 自动记录参数、指标、混淆矩阵和模型。

## 常用查询命令

```bash
mlflow runs list --experiment-name "KNN-LR实验追踪演示"
```

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
exp = client.get_experiment_by_name("KNN-LR实验追踪演示")
runs = client.search_runs(
    [exp.experiment_id],
    order_by=["metrics.accuracy DESC"],
)
print(runs[0].info.run_id, runs[0].data.metrics)
```

## 注意事项

- `mlflow.db` 保存实验、Run、参数和指标，`mlruns/` 保存模型与图片等制品；两者都不要手动修改或删除。
- `artifacts_out/` 只是本地产物缓存，上传后在 MLflow 网页的 Artifacts 中仍可查看。
- 如果网页里看不到实验，先确认启动 `mlflow ui` 时终端位于 `mlflow_demo` 目录，并且 `--backend-store-uri` 指向同一个 `mlflow.db`。
- MLflow 3.x 把模型放在独立的 **Models** 区域（Logged Models），但 `runs:/<run_id>/knn_model` 仍然可以加载模型。