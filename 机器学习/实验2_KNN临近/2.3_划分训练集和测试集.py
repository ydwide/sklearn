import numpy as np
from sklearn.datasets import load_iris

#1.加载数据，了解数据
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state = 25)

print(X_test,y_test)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
