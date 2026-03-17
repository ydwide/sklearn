import numpy as np
from sklearn.datasets import load_iris #加载鸢尾花数据集

# 1.加载数据，了解数据
iris = load_iris()
X = iris.data
print(X.shape)
print(X[0:5,:])#前五行，全部列
print(iris.feature_names)

y = iris.target
print(y[:5]) #y是标签，是个向量，只有一个维度
print(y)  #0,1,2表示三个类别
print(np.bincount(y)) #查看类别中有几条数据,查看类别标签分布情况
print(iris.target_names) #查看类别名称
