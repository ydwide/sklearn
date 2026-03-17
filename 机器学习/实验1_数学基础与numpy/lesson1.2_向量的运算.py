import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])
c = a + b
print(c)

d = 3
print(a + d) # 向量+标量
print(a * d) # 向量*标量
print(a * b) # 向量*向量    asd

# b2 = np.array([1,2,3,4])
# print(a * b2) # 向量元素个数不统一无法相乘

e = np.dot(a,b)  # 内积（点积）
print(e)
f = a @ b
print(f)