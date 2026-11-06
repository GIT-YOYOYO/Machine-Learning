# ===== 导入所需模块 =====
# 导入os模块，用来拼接文件路径、创建输出文件夹
import os
# 导入mlflow主库，用于实验追踪
import mlflow
# 导入mlflow的sklearn专用模块，用来保存、加载sklearn模型
import mlflow.sklearn
# 导入鸢尾花数据集，用于KNN分类
from sklearn.datasets import load_iris
# 导入数据集划分工具
from sklearn.model_selection import train_test_split
# 导入K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入分类评估指标：准确率、混淆矩阵
from sklearn.metrics import accuracy_score, confusion_matrix
# 导入绘图库，用来生成混淆矩阵图片
import matplotlib.pyplot as plt

# 设置中文字体，保证图表标题和坐标标签正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 准备输出目录 =====
# 定义本地输出文件夹名
out_dir = "artifacts_out"
# 若文件夹不存在则自动创建，防止学生报"找不到路径"错误
os.makedirs(out_dir, exist_ok=True)

# 设置实验分组名称，后续所有run都会归类到"KNN-LR实验追踪演示"这个实验下
mlflow.set_experiment("KNN-LR实验追踪演示")

# 用with开启一次独立Run（run_name给本次实验起名，方便网页识别）
with mlflow.start_run(run_name="KNN第一次手动记录run") as run:
    # ========== 动作1：log_param 记录超参 ==========
    # 定义KNN的超参K值（邻近样本数量）
    k = 5
    # log_param(参数名, 参数值)：记录静态超参，训练过程不会改变
    mlflow.log_param("n_neighbors", k)

    # ===== 数据加载与划分 =====
    # 加载鸢尾花数据集：X是特征，y是标签
    X, y = load_iris(return_X_y=True)
    # 划分训练集、测试集：测试集占20%，固定随机种子42保证可复现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===== 模型训练与预测 =====
    # 实例化KNN分类器，传入K值
    knn_model = KNeighborsClassifier(n_neighbors=k)
    # 使用训练集拟合训练模型
    knn_model.fit(X_train, y_train)
    # 在测试集上预测
    y_pred = knn_model.predict(X_test)
    # 计算测试集准确率
    acc = accuracy_score(y_test, y_pred)

    # ========== 动作2：log_metric 记录评估指标 ==========
    # log_metric(指标名, 指标值)：记录模型效果分数
    mlflow.log_metric("accuracy", acc)

    # ========== 动作3：log_artifact 保存图片制品 ==========
    # 计算混淆矩阵：真实标签 vs 预测标签
    cm = confusion_matrix(y_test, y_pred)
    # 拼接图片的完整保存路径
    img_path = os.path.join(out_dir, "knn_conf_matrix.png")
    # 创建画布
    plt.figure()
    # 绘制混淆矩阵热力图
    plt.imshow(cm, cmap=plt.cm.Blues)
    # 设置图片标题
    plt.title("KNN混淆矩阵")
    # 添加右侧颜色条
    plt.colorbar()
    # 保存图片到本地artifacts_out文件夹
    plt.savefig(img_path)
    # 关闭画布释放内存，防止多次绘图冲突
    plt.close()
    # 将图片上传绑定到当前run，网页界面可直接点开查看
    mlflow.log_artifact(img_path)

    # ========== 动作4：log_model 保存模型制品 ==========
    # 将sklearn模型打包保存为mlflow制品，后续可直接加载预测
    mlflow.sklearn.log_model(
		sk_model=knn_model,
		name="knn_model",
		skops_trusted_types=[
			"sklearn.metrics._dist_metrics.EuclideanDistance64",
			"sklearn.neighbors._kd_tree.KDTree"
		]
	)

    # 打印本次run唯一编号，方便后续查找
    print(f"本次run id: {run.info.run_id}")
    # 打印准确率，保留4位小数
    print(f"K={k}，准确率：{acc:.4f}")
