import numpy as np

x = np.array([3,-4,0])
y = np.array([0,0,0])
print(np.linalg.norm(x, ord=2))  # l2范数 欧几里得范数 模长
print(np.linalg.norm(x, ord=1))  # l1范数 绝对值范数   各元素绝对值相加


a = x - y
print(np.linalg.norm(a, ord=2))  # 两个向量的欧式（l2）距离
print(np.linalg.norm(a, ord=1))  # 2个向量的l1距离

l2_manual = np.sqrt(np.sum(a**2)) # 数学公式手动算值
l1_manual = np.sum(np.abs(a))  #l1值
print(l2_manual, l1_manual)