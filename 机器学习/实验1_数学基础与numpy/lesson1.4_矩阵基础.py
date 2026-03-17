import numpy as np

a = np.array([
    [1,2,3],
    [4,5,6],
])
print(a)
print(a.shape)  # (2,3)两行三列

b = np.arange(9).reshape((3, 3))  # 生成等差数列的数组 np.arange(start,stop,step,dtype=None)(起始，结束（不包含），步长(支持小数步长)，数据类型)
print(b)

c = np.eye(3,dtype=int)  #单位矩阵（对角线全部为1） np.eye(n,dtype) n为元素个数，dtype为元素类型，默认为浮点数
print(c)

d = np.diag([1,2,3])  # 对角矩阵(除了对角线上的元素其他都为零)
print(d)

e = np.diag(b)  # 用diag取一个方阵的对角线数组
print(e)