import numpy as np

a = np.arange(6).reshape(2,3)
print(a)
print(a.T)  # 矩阵转置


#求b的逆矩阵
b = np.array([
    [2, 1],
    [5, 3]
])
b_inv = np.linalg.inv(b)
print(b)
print(b_inv)  # 逆矩阵与原矩阵相乘等于单位矩阵

I = np.eye(2)
print(I)
print(np.round((b @ b_inv),6))  #浮点数存不准
print(np.allclose(I, np.eye(2))) 