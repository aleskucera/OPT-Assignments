import numpy as np
from scipy.optimize import lsq_linear
from scipy import linalg


# noinspection PyPep8Naming
class AutoregressiveModel(object):
    def __init__(self, data: np.array, order: int):
        self.y = data
        self.p = order
        self.M = self._init_matrix()
        self.a = self._ar_fit_model()
        self._print_status()

    def _print_status(self):
        print(f"INFO: Model created.")
        print(f"INFO: Shape of given data (y): {self.y.shape}")
        print(f"INFO: Model order (p): {self.p}")
        print(f"INFO: Matrix shape (M): {self.M.shape}")
        print(f"INFO: Parameters shape (a): {self.a.shape}")

    def _init_matrix(self) -> np.array:
        print(f"INFO: Creating matrix (M)...")
        T = self.y.shape[0]
        matrix_shape = (T - self.p, self.p)
        matrix = np.ones(matrix_shape)
        with np.nditer(matrix, flags=["multi_index"], op_flags=["writeonly"]) as it:
            for x in it:
                t = it.multi_index[0] + self.p - 1  # not sure yet
                i = it.multi_index[1]
                if i != 0:
                    x[...] = self.y[t - i]
        return matrix

    def _ar_fit_model(self):
        print(f"INFO: Looking for optimal parameters...")
        res = lsq_linear(self.M, self.y[self.p:])
        if res.success:
            print(f"INFO: Successfully found optimal parameters for autoregressive model.")
            print(f"INFO: Total cost is {res.cost}.")
        else:
            print(f"ERROR: Problem occurred in finding optimal parameters.")
        return res.x

    def ar_predict(self, N: int):
        M = self.M[:N, :]
        seq = M @ self.a
        seq = np.concatenate((self.y[:self.p], seq))
        print(f"INFO: Successfully predicted sequence of shape {seq.shape}.")
        return seq

    @staticmethod
    def solve_ls(A: np.array, b: np.array) -> np.array:
        Q, R = linalg.qr(A, mode="economic")
        x = linalg.solve_triangular(R, Q.T.dot(b))
        return x
