#步骤1.导入实验环境与库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#步骤2：加载数据集并查看特征
data = load_diabetes()
for idx,name in enumerate(data.feature_names):
    print(idx, name)

#步骤3：提取实验变量与目标特征
X_original = data.data[:,2].reshape(-1,1)
y_original = data.target.reshape(-1,1)

#步骤4：构建数据表
data_pd = pd.DataFrame({
    'BMI':X_original.flatten(),
    '疾病进程':y_original.flatten()
})
print(data_pd.describe())
print(data_pd.head())

'''
#步骤5：数据可视化
fig, ax =plt.subplots(figsize=(12,8))
ax.scatter(data_pd['BMI'], data_pd['疾病进程'])
ax.set_xlabel('BMI')
ax.set_ylabel('疾病进程')
plt.show()
'''

#步骤6：构建特征矩阵并加入偏置项
data_pd.insert(loc=0,column='Ones',value=1)
print(data_pd.head())

X = np.array(data_pd.iloc[:,:-1])
y = np.array(data_pd.iloc[:,-1:])
print(X.shape)
print(y.shape)

#步骤7：实现代价函数
def computerCost(X, y, w):
    inner = np.power(X @ w - y, x2=2) # 幂运算（2就是平方）
    out = np.sum(inner) / (2*X.shape[0]) #这里样本的数量是442
    return out

#步骤8：使用最小二乘法求解参数
def LSM(X,y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w
optimal_w = LSM(X,y)
print('最优参数w：',optimal_w)

#步骤9：模型预测与回归线绘制
y_pred = X @ optimal_w
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,1],y,label = '训练数据')
ax.plot(X[:,1],y_pred,label='回归直线')
ax.set_xlabel("BMI")
ax.set_ylabel("疾病进程")
ax.legend()
plt.show()