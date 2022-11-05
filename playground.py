import math
import numpy as np

def my_newton(f, df, x: np.ndarray, epsilon: float, max_iter: int) -> np.ndarray:
    if np.sqrt(np.sum(x[0] + x[1])) < epsilon or max_iter == 0:
        return x
    else:
        return my_newton(f, df, x - f(x)/df(x), epsilon, max_iter - 1)

def f(x: np.ndarray) -> float:
    return x[0]**2 - x[1] - 1 + np.sin(x[1]**2 - 2*x[0])

def df(x: np.ndarray) -> np.ndarray:
    return np.array([2*x[0] + 2*np.cos(x[1]**2 - 2*x[0]), -1 - 2*x[1]*np.cos(x[1]**2 - 2*x[0])])

x = np.array([1, 1])
epsilon = 1e-4
max_iter = 500

print(my_newton(f, df, x, epsilon, max_iter))
