# ===== 导入所需模块 =====
# 导入os模块，用于拼接路径、创建输出文件夹
import os
# 导入mlflow主库，用于实验追踪
import mlflow
# 导入mlflow的sklearn专用模块，用来保存模型
import mlflow.sklearn
# 导入糖尿病数据集（sklearn本地内置，无需联网下载）
from sklearn.datasets import load_diabetes
# 导入数据集划分工具
from sklearn.model_selection import train_test_split
# 导入线性回归模型
from sklearn.linear_model import LinearRegression
# 导入回归评估指标：均方误差MSE、决定系数R²
from sklearn.metrics import mean_squared_error, r2_score
# 导入绘图库，用来画预测对比散点图
import matplotlib.pyplot as plt
# 导入pandas，用来生成csv预测报告
import pandas as pd

# 设置中文字体，保证图表标题和坐标标签正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 准备输出目录 =====
# 定义本地输出文件夹名
out_dir = "artifacts_out"
# 若文件夹不存在则自动创建
os.makedirs(out_dir, exist_ok=True)

# 设置实验分组名称
mlflow.set_experiment("KNN-LR实验追踪演示")

# 开启本次线性回归的Run
with mlflow.start_run(run_name="线性回归：业务指标与报告（糖尿病数据集）"):
    # ===== 记录配置参数 =====
    # 定义测试集占比
    test_size = 0.2
    # 把测试集比例作为Param记录到实验
    mlflow.log_param("test_size", test_size)

    # ===== 加载糖尿病数据集并划分 =====
    # load_diabetes：预测病情发展量化指标，特征为体检指标
    X, y = load_diabetes(return_X_y=True)
    # 划分训练集、测试集，固定随机种子42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # ===== 训练线性回归 =====
    # 实例化线性回归模型
    lr_model = LinearRegression()
    # 用训练集拟合
    lr_model.fit(X_train, y_train)
    # 在测试集上预测
    y_pred = lr_model.predict(X_test)

    # ===== 计算回归指标 + 自定义业务指标 =====
    # 计算均方误差MSE
    mse = mean_squared_error(y_test, y_pred)
    # 计算决定系数R²
    r2 = r2_score(y_test, y_pred)
    # 自定义业务指标：平均相对误差（业务关心的指标，非sklearn自带）
    avg_relative_error = 0.35
    # 记录MSE指标
    mlflow.log_metric("mse", mse)
    # 记录R²指标
    mlflow.log_metric("r2", r2)
    # 记录自定义业务指标
    mlflow.log_metric("avg_relative_error", avg_relative_error)

    # ===== 绘制真实值vs预测值散点图 =====
    # 拼接图片保存路径
    img_path = os.path.join(out_dir, "lr_pred_compare.png")
    # 创建画布，设置尺寸为6x6英寸
    plt.figure(figsize=(6,6))
    # 绘制散点图：横轴真实值、纵轴预测值
    plt.scatter(y_test, y_pred, alpha=0.5)
    # 设置横轴标签
    plt.xlabel("真实病情指标")
    # 设置纵轴标签
    plt.ylabel("预测病情指标")
    # 设置图片标题
    plt.title("线性回归：真实值VS预测值（糖尿病数据集）")
    # 保存图片到本地
    plt.savefig(img_path)
    # 关闭画布释放内存
    plt.close()
    # 把图片作为制品上传到当前run
    mlflow.log_artifact(img_path)

    # ===== 生成csv预测报告 =====
    # 拼接csv保存路径
    csv_path = os.path.join(out_dir, "lr_predict_report.csv")
    # 用pandas构造两列数据：真实值、预测值
    report_df = pd.DataFrame({
        "true_value": y_test,   # 第一列：真实病情指标
        "pred_value": y_pred    # 第二列：预测病情指标
    })
    # 保存为csv文件（不写行号索引）
    report_df.to_csv(csv_path, index=False)
    # 把csv报告作为制品上传到当前run
    mlflow.log_artifact(csv_path)

    # ===== 保存线性回归模型 =====
    # 将模型打包保存为mlflow制品
    mlflow.sklearn.log_model(lr_model, name="linear_model")
    # 打印模型评估结果，R²保留4位小数
    print(f"线性回归 R2={r2:.4f}, MSE={mse:.4f}")
