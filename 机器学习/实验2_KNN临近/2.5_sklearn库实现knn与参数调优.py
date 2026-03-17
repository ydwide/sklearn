import numpy as np
import sklearn
from sklearn.datasets import load_iris
from collections import Counter

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state = 1)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

from sklearn.neighbors import KNeighborsClassifier
#1.用sklearn库，实现KNN的预测
knn = KNeighborsClassifier(n_neighbors = 3,metric = 'euclidean') #使用欧式距离
knn.fit(X_train, y_train) # 训练，KNN只是存储训练数据，KNN不需要训练的
y_pred = knn.predict(X_test)
acc = accuracy(y_test, y_pred)
print(acc)

#2.网络搜索调优k值
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': range(1,16)}
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy') #cv N折交叉验证
grid.fit(X_train, y_train)
print('最优K值',{grid.best_params_['n_neighbors']})
print('最优交叉验证准确度',{grid.best_score_})

#3.使用最优K值构建模型
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)
print(accuracy(y_test, y_pred))

