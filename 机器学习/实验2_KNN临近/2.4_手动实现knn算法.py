import numpy as np
from sklearn.datasets import load_iris
from collections import Counter

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state = 1)

#1.欧式距离计算
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

#2.实现KNN预测函数
def knn_predict(X_train, y_train, X_test, k):
    distances = [euclidean_distance(X_test,x_train_i) for x_train_i in X_train]
    k_indices = np.argsort(distances)[:k] # 排序，取前k个值
    k_neighbors_labels = [y_train[i] for i in k_indices] # 取出对应的label值
    return Counter(k_neighbors_labels).most_common(1)[0][0]
'''
    counter统计列表中每个元素出现的次数，most_common(1)返回出现次数最多的前1个元素，
    格式为列表装元组[(1,2)],1表示类别，2表示出现次数，[0][0]即列表第一个元素(即唯一的元组)的第一个元素(即类别)
'''
#3.对测试集所有样本进行预测
def knn_classify(X_train,y_train,Xtest,k):
    predictions = [knn_predict(X_train,y_train,x,k) for x in Xtest]
    return np.array(predictions)

#4计算准确率
def accuracy(y_true,y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
'''
y_true = [0 0 1 2 1]
y_pred = [1 0 1 1 1]
acc    = [0 1 1 0 1]
sum    = 3
len    = 5
acc    = 3/5=0.6
'''

k = 3
y_pred = knn_classify(X_train,y_train,X_test,k)
acc = accuracy(y_test,y_pred)
print(acc)

#尝试不同k值
for k in [1,3,9,10,20,40,90]:
    y_pred = knn_classify(X_train,y_train,X_test,k)
    acc = accuracy(y_test,y_pred)
    print(k,acc)