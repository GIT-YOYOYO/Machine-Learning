# ===== 导入所需模块 =====
# 导入mlflow主库，用于实验追踪
import mlflow
# 导入MlflowClient客户端，用来用代码查询历史实验
from mlflow.tracking import MlflowClient
# 导入鸢尾花数据集，用于最后预测演示
from sklearn.datasets import load_iris

# 设置实验分组名称（必须和前面运行实验时一致，才能查到那些run）
mlflow.set_experiment("KNN-LR实验追踪演示")

# ===== 创建客户端并定位实验 =====
# 创建mlflow客户端对象
client = MlflowClient()
# 根据实验名称找到对应的实验对象
exp = client.get_experiment_by_name("KNN-LR实验追踪演示")
# 获取实验id，查询时需要用到
exp_id = exp.experiment_id

# ===== 查询所有run，按准确率降序排序 =====
runs = client.search_runs(
    experiment_ids=[exp_id],      # 指定要查询的实验id
    filter_string="",             # 不添加额外筛选条件，查全部
    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,  # 只看激活状态（未删除）的run
    order_by=["metrics.accuracy DESC"]  # 按准确率降序排列，效果最好排第一
)

# ===== 取出最优run =====
# 取排序第一，即效果最好的run
best_run = runs[0]
# 打印最优实验的标题
print("===== 最优KNN实验结果 =====")
# 打印最优run的唯一编号
print(f"run_id: {best_run.info.run_id}")
# 打印最优run的参数信息
print(f"参数信息: {best_run.data.params}")
# 打印最优准确率，保留4位小数
print(f"最优准确率: {best_run.data.metrics['accuracy']:.4f}")

# ===== 加载最优模型 =====
# 根据run_id拼接模型存储地址（runs:/格式是mlflow标准模型引用）
best_model_uri = f"runs:/{best_run.info.run_id}/knn_model"
# 从该地址加载训练好的KNN模型
best_knn_model = mlflow.sklearn.load_model(best_model_uri)

# ===== 用最优模型做预测演示 =====
# 重新加载鸢尾花数据集的样本
X, y = load_iris(return_X_y=True)
# 用最优模型预测前5个样本
print("最优KNN模型预测样例：", best_knn_model.predict(X[:5]))
