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

