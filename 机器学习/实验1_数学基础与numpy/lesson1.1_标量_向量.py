import numpy as np

#a = 4
a2 = np.array(4) # 用np定义的一个标量
b = np.array([1,3,6])  #一个向量

print (a2)
print (b)
print (a2.ndim)
print (b.ndim)
print (a2.shape)
print (b.shape)  # (3,) 3表示元素个数
print (b.T)

c = b.reshape(1,-1) # 把向量转成 1行 *列的矩阵
print (c)
d = b.reshape(-1,1) # 把向量转成 *行 1列的矩阵
print (d)
#print (c.T)

print (c.ndim)
print (c.shape)
print (d.ndim)
print (d.shape)

print ('(3,)表示1维3个元素的向量,(1,3)表示二维1行3列，(3,1)表示二维三行一列')