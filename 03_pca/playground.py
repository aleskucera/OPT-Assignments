import numpy as np

# create random 10x10 matrixs
A = np.random.randint(high=10, low=1, size=(10, 3))
k = 2
A[k:, k:] = 0
print(A)