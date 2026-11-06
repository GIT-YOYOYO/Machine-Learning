# ===== 导入所需模块 =====
# 导入os模块，用于创建输出目录
import os
# 导入mlflow主库，用于实验追踪
import mlflow
# 导入mlflow的sklearn专用模块，用来保存模型
import mlflow.sklearn
# 导入鸢尾花数据集，用于逻辑回归分类
from sklearn.datasets import load_iris
# 导入数据集划分工具
from sklearn.model_selection import train_test_split
# 导入逻辑回归模型
from sklearn.linear_model import LogisticRegression
# 导入准确率评估指标
from sklearn.metrics import accuracy_score

# ===== 准备输出目录 =====
# 定义本地输出文件夹名
out_dir = "artifacts_out"
# 若文件夹不存在则自动创建
os.makedirs(out_dir, exist_ok=True)

# 设置实验分组名称
mlflow.set_experiment("KNN-LR实验追踪演示")

# ===== 核心：开启自动记录 =====
# autolog：一行代码开启自动记录，自动抓取参数、评估指标、混淆矩阵、模型，无需手写log
mlflow.autolog()

# 开启本次逻辑回归的Run
with mlflow.start_run(run_name="autolog自动记录_逻辑回归"):
    # 定义逻辑回归的正则系数C
    c = 1.0
    # 加载鸢尾花数据集：X特征、y标签
    X, y = load_iris(return_X_y=True)
    # 划分训练集、测试集，测试集占20%，固定随机种子
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 实例化逻辑回归，max_iter加大保证收敛，C控制正则化强度
    logreg = LogisticRegression(C=c, max_iter=200)
    # 用训练集拟合模型
    logreg.fit(X_train, y_train)
    # 在测试集上预测
    y_pred = logreg.predict(X_test)
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    # 打印结果，保留4位小数
    print(f"逻辑回归 C={c}，准确率：{acc:.4f}")

# 运行后打开网页查看：参数、准确率、混淆矩阵、模型已被自动记录
