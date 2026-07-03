import numpy as np
from collections.abc import Callable

class SVM:
    def __init__(self, C: float = 1.0, seed: int = 42):
        self.w = None
        self.C = C
        self.fitted = False
        self.b = 0
        self._rng = np.random.default_rng(seed=seed)

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        if C <= 0:
            raise ValueError("C must be a positive value.")
        self._C = C

    def _validate_train_set(self, X, y):
        unique = np.unique(y)
        if not (len(unique) == 2 and unique[0] == -1 and unique[1] == 1):
            raise ValueError("y must contain only -1 and 1.")

        if not X.shape[0] == len(y):
            raise ValueError(f"X and y must have the same number of samples. Got:"
                             f"{X.shape[0]} for X and {len(y)} for y.")

    def fit(self, X, y, lr: float = 1e-4, epochs: int = 100, verbose: bool = False):
        self._validate_train_set(X, y)
        samples, features = X.shape
        self.w = np.zeros(shape=(features,))
        self.b = 0
        self.loss: list[float] = []
        for epoch in range(1, epochs + 1):
            for sample_idx in self._rng.permutation(samples):
                margin = y[sample_idx] * (np.dot(self.w, X[sample_idx]) + self.b)
                if margin >= 1:
                    dw = self.w / samples
                    db = 0

                else:
                    dw = self.w / samples - self._C * y[sample_idx] * X[sample_idx]
                    db = - self._C * y[sample_idx]

                self.w -= lr * dw
                self.b -= lr * db

            self.loss.append(self.loss_function(X, y))
            if verbose:
                print(f"Epoch: {epoch} | Loss: {self.loss_function(X, y)}")

        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model must be fit before calling predict.")
        return np.sign(X @ self.w + self.b)
    
    def loss_function(self, X, y):
        self._validate_train_set(X, y)
        margins = y * (X @ self.w + self.b)
        hinge_loss = np.sum(np.maximum(0, 1 - margins))
        return 0.5 * np.dot(self.w, self.w) + self._C * hinge_loss

    def support_vectors(self, X, y, tol=1e-4):
        if not self.fitted:
            raise RuntimeError("Model must be fit before calling support vectors.")
        self._validate_train_set(X, y)
        margins = y * (X @ self.w + self.b)
        return np.where(margins <= 1 + tol)[0]
    
    def decision_function(self, X):
        if not self.fitted:
            raise RuntimeError("Model must be fit before calling decision function.")
        return X @ self.w + self.b

class Kernel(Callable):
    def __call__(self, x, y):
        raise NotImplementedError

class LinearKernel(Kernel):
    def __call__(self, x: np.ndarray, y: np.ndarray):
        return x @ y.T
    
class RBFKernel(Kernel):
    def __init__(self, gamma: float = 1):
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(x)
        Y = np.atleast_2d(y)
        sq = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)  # (nx, ny)
        out = np.exp(-self.gamma * sq)
        if y.ndim == 1:      # kernel(matrix, vector) -> (nx,)
            out = out[:, 0]
        if x.ndim == 1:      # kernel(vector, ...) -> drop first axis
            out = out[0]
        return out
    
class PolynomialKernel(Kernel):
    def __init__(self, c: float = 1, d: int = 2):
        self.c = c
        self.d = d

    def __call__(self, x: np.ndarray, y: np.ndarray):
        return (x @ y.T + self.c) ** self.d


class KernelSVM:
    def __init__(self, C: float = 1.0, kernel: Kernel = None, seed: int = 42):
        self.C = C
        self.fitted = False
        self.b = 0
        self.kernel = kernel
        self._rng = np.random.default_rng(seed=seed)

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        if C <= 0:
            raise ValueError("C must be a positive value.")
        self._C = C

    def _validate_train_set(self, X, y):
        unique = np.unique(y)
        if not (len(unique) == 2 and unique[0] == -1 and unique[1] == 1):
            raise ValueError("y must contain only -1 and 1.")

        if not X.shape[0] == len(y):
            raise ValueError(f"X and y must have the same number of samples. Got:"
                             f"{X.shape[0]} for X and {len(y)} for y.")

    def fit(self, X, y, lr: float = 1e-4, epochs: int = 100, sv_tol: float = 1e-3):
        self._validate_train_set(X, y)
        samples, features = X.shape

        alpha = np.zeros(samples)
        b = 0
        for epoch in range(1, epochs + 1):
            for sample_idx in self._rng.permutation(samples):
                s = np.sum(alpha*y*self.kernel(X, X[sample_idx])) + b
                if y[sample_idx] * s < 1:
                    alpha[sample_idx] = min(alpha[sample_idx] + lr*self._C, self._C)
                    b += lr*self._C*y[sample_idx]


            alpha *= 1 - lr / samples

        self.b = b
        self.alpha = alpha

        threshold = sv_tol * alpha.max() if alpha.max() > 0 else 0.0
        self._support = alpha > threshold
        self._support_vectors = X[self._support]
        self._support_labels = y[self._support]

        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Classifier should be fit before predict")
        K = self.kernel(self._support_vectors, X)          # (m_sv, k) or (m_sv,)
        s = (self.alpha[self._support] * self._support_labels) @ K
        return np.sign(s + self.b)

    def decision_function(self, X):
        if not self.fitted:
            raise RuntimeError("Classifier should be fit before predict")
        K = self.kernel(self._support_vectors, X)          # (m_sv, k) or (m_sv,)
        s = (self.alpha[self._support] * self._support_labels) @ K
        return s




