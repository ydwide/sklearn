import numpy as np
from sklearn.datasets import load_iris #加载鸢尾花数据集
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
plt.scatter(X[:,2],X[:,3],c=y)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('petal length and petal width')
plt.show()