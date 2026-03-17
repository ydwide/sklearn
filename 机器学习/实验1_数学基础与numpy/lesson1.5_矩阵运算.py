import numpy as np

A = np.arange(6).reshape(2,3)
B = np.arange(6).reshape(2,3)
C = np.arange(12).reshape(3,4)
print(A)
print(B)

print(A+B) #矩阵形状不一样不能相加

print(2 * A)

print(B @ C) # 矩阵相乘，要求前一个矩阵的列数与后一个矩阵的行数相同
             # 2，3  和   3，4  得到 2，4 矩阵
