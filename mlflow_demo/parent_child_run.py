# ===== 导入所需模块 =====
# 导入os模块，用于创建输出目录
import os
# 导入mlflow主库，用于实验追踪
import mlflow
# 导入mlflow的sklearn专用模块，用来保存模型
import mlflow.sklearn
# 导入鸢尾花数据集，用于KNN分类
from sklearn.datasets import load_iris
# 导入数据集划分工具
from sklearn.model_selection import train_test_split
# 导入K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入准确率评估指标
from sklearn.metrics import accuracy_score

# ===== 准备输出目录 =====
# 定义本地输出文件夹名
out_dir = "artifacts_out"
# 若文件夹不存在则自动创建
os.makedirs(out_dir, exist_ok=True)

# 设置实验分组名称
mlflow.set_experiment("KNN-LR实验追踪演示")

# ===== 开启父Run：代表整组批量调参任务 =====
with mlflow.start_run(run_name="【父run】KNN批量搜索K值") as parent_run:
    # 获取父run的唯一id，给子run绑定，建立父子关系
    parent_id = parent_run.info.run_id
    # 需要遍历测试的K值列表
    k_list = [1, 3, 5, 7]

    # 循环遍历每个K值，每一个K开启一个独立子Run
    for k in k_list:
        # 开启子Run，传入parent_run_id，在网页UI里会折叠挂在父Run下面
        with mlflow.start_run(run_name=f"子run K={k}", parent_run_id=parent_id, nested=True):
            # 记录当前子Run的K参数
            mlflow.log_param("n_neighbors", k)
            # 加载鸢尾花数据集：X特征、y标签
            X, y = load_iris(return_X_y=True)
            # 划分训练集、测试集，测试集占20%，固定随机种子
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # 实例化KNN分类器，传入当前K值
            knn = KNeighborsClassifier(n_neighbors=k)
            # 用训练集拟合
            knn.fit(X_train, y_train)
            # 在测试集上预测
            y_pred = knn.predict(X_test)
            # 计算准确率
            acc = accuracy_score(y_test, y_pred)
            # 记录当前子Run的准确率指标
            mlflow.log_metric("accuracy", acc)
            # 保存当前K对应的模型
            mlflow.sklearn.log_model(
				sk_model=knn,
				name="knn_model",
				input_example=X_train[:5],
				skops_trusted_types=[
					"sklearn.metrics._dist_metrics.EuclideanDistance64",
					"sklearn.neighbors._kd_tree.KDTree"
				]
			)
            # 打印当前子实验的结果
            print(f"子实验 K={k}，准确率={acc:.4f}")
