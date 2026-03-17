import numpy as np

#1 构造矩阵
A = np.array([
    [1,3,7],
    [2,5,10]
])
B = np.array([
    [2,4,7],
    [4,8,14]
])

#2 计算矩阵秩
a_rank = np.linalg.matrix_rank(A)
b_rank = np.linalg.matrix_rank(B)
print (a_rank)
print (b_rank)
b_range = B.shape[0]  #矩阵行数
if b_rank < b_range:
    print ("存在线性相关行，秩小于矩阵行数")

#3 计算矩阵Frobenius范数，并人工验证 ,即F范数，矩阵每个元素平方之和的平方根，与向量L2范数类似，可用于衡量矩阵整体大小
fro1 = np.linalg.norm(A,ord='fro')
fro2 = np.linalg.norm(B,ord='fro')
print (fro1)
print (fro2)

#3.1人工验证
fro3 = np.sqrt(np.sum(A**2))
print (fro3)
print (np.allclose(fro1,fro3))