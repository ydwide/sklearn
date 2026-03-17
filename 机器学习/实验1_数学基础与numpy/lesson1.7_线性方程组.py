import numpy as np

'''
2x1 + x2 = 1
5x1 + 3x2 = 2
'''

# 1
A = np.array([
    [2,1],
    [5,3]
])
b = np.array([1,2])

#2
x = np.linalg.solve(A,b)
print(x)

#3
c = (A @ x)
print(np.allclose(c,b))