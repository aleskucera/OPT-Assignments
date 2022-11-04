import numpy as np

p = 3
y = np.array([1,2,3,4,5,6,7,8,9,10])

m = y.shape[0] - p
n = p + 1
M = np.ones((m, n))
for i in range(0, m):
    M[i, 1:] = y[i:i+p]

print(y[p:])
a = np.linalg.lstsq(M, y[p:], rcond=None)[0]

# M = np.ones((y.shape[0]-p, p+1))
# for i in range(p, y.shape[0]):
#     M[i-p,1:] = y[i-p:i]
#
# a = np.linalg.lstsq(M, y[p:])[0]

print(M.shape)
print(M)
print(a)
