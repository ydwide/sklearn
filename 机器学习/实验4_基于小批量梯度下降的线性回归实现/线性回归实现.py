#步骤1 导入实验环境
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler #用于数据标准化

#步骤2 加载数据集并提取特征
diabetes = load_diabetes()
X = diabetes.data[:,2].reshape(-1,1) #BMI
y = diabetes.target.reshape(-1,1) #疾病进程

#步骤3 数据标准化 z-score 标准化， 要算平均值和标准差
scalarX = StandardScaler()
X = scalarX.fit_transform(X)
scalery = StandardScaler()
y = scalery.fit_transform(y)

print(X[:5,])
print(y[:5,])

# 步骤4 构建特征矩阵，加入偏置项
X = np.hstack([np.ones((X.shape[0],1)),X])
print(X[:5,])

# 步骤5 实现Mini-Batch梯度下降
# lr: 学习率
# epochs: 训练批次
# barch_size: 小批量的样本量
def MBGD(X, y, lr=0.05, epochs=100, batch_size=16):
    m, n = X.shape # 行数， 列数
    theta = np.zeros((n,1)) #初始w值
    loss_history =[]
    for epoch in range(epochs):
        indices = np.random.permutation(m) #shuffle数据，随机打乱样本顺序
        x = X[indices]
        y = y[indices]

        for i in range(0, m, batch_size):
            Xb = X[i:1 + batch_size]
            yb = y[i:1 + batch_size]
            y_pred = Xb @ theta
            grad = (Xb.T @ (y_pred-yb)) / len(Xb)

            theta -= lr @ grad #更新