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

#步骤5：数据可视化
fig, ax =plt.subplots(figsize=(12,8))
ax.scatter(data_pd['BMI'], data_pd['疾病进程'])
ax.set_xlabel('BMI')
ax.set_ylabel('疾病进程')
plt.show()