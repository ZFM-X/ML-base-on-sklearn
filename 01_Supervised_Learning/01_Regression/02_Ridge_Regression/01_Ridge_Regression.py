import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams["axes.unicode_minus"] = False

# 1. 导入所需的库
# 已在代码开头导入

# 2. 加载回归数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 3. 数据预处理
# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 特征工程
# 在这个例子中，我们直接使用了原始特征，没有进行额外的特征工程

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 6. 定义岭回归模型和参数网格
model = Ridge()
param_grid = {
    'alpha': [0.1, 1, 10, 100]  # 正则化强度
}

# 使用网格搜索，寻找最佳参数（包含训练得分）
grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

# 7. 网格搜索后的最佳参数和模型
print("最佳参数：", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 8. 单独打印训练集和测试集的模型评估指标
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)
print(f"训练集 RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

# 9. 绘制模型的训练曲线
cv_results = grid_search.cv_results_
mean_train_score = cv_results['mean_train_score']
mean_test_score = cv_results['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(mean_train_score, label='训练得分')
plt.plot(mean_test_score, label='测试得分')
plt.title('模型性能曲线')
plt.xlabel('参数组合编号')
plt.ylabel('得分')
plt.legend()
plt.show()

# 10. 可视化真实值与预测值的关系
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('真实值 vs 预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.show()

# 11. 绘制误差分布图
errors = y_test - test_predictions
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.xlabel('预测误差')
plt.ylabel('频率')
plt.title('误差分布图')
plt.show()
