from abc import ABC, abstractmethod
import numpy as np

EPS = 1e-5

class Activation(ABC):
    @staticmethod
    @abstractmethod
    def forward(z):
        ...

    @staticmethod
    @abstractmethod
    def gradient(z):
        ...


class Sigmoid(Activation):
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        a = Sigmoid.forward(z)
        return a * (1 - a)

class ReLU(Activation):
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def gradient(z):
        return (z > 0).astype(z.dtype)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, random_seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.random_seed = random_seed

    def fit(self, X, y, lr=1e-3, n_iterations=1000):
        if np.unique(y).shape[0] != 2:
            raise ValueError("y must be binary")
        
        self.classes_ = np.unique(y)
        y = np.where(y == self.classes_[0], 0, 1).astype(np.float64)

        self._rng = np.random.default_rng(self.random_seed)
        self._w1 = self._rng.normal(0, 1, (self.input_size, self.hidden_size))
        self._b1 = np.zeros(self.hidden_size)
        self._w2 = self._rng.normal(0, 1, (self.hidden_size, self.output_size))
        self._b2 = np.zeros(self.output_size)

        self.loss_history_ = []
        for iter in range(1, n_iterations + 1):
            z1, a1, z2, a2 = self._forward(X)
            self.loss_history_.append(self._loss(y, a2))
            dw1, db1, dw2, db2 = self._backward(X, y, z1, a1, z2, a2)

            self._w1 -= lr * dw1
            self._b1 -= lr * db1
            self._w2 -= lr * dw2
            self._b2 -= lr * db2

    def _loss(self, y, a2):
        y = y.reshape(-1, 1)
        return -np.mean(y * np.log(a2 + EPS) + (1 - y) * np.log(1 - a2 + EPS))

    def _forward(self, X):
        z1 = X @ self._w1 + self._b1  # (samples, hidden_size)
        a1 = ReLU.forward(z1)         # (samples, hidden_size)
        z2 = a1 @ self._w2 + self._b2 # (samples, output_size)
        a2 = Sigmoid.forward(z2)      # (samples, output_size)

        return z1, a1, z2, a2
    
    def _backward(self, X, y, z1, a1, z2, a2):
        n_samples = X.shape[0]

        dz2 = a2 - y.reshape(-1, 1)            # (samples, output_size)
        dw2 = a1.T @ dz2 / n_samples           # (hidden_size, output_size)
        db2 = np.sum(dz2, axis=0) / n_samples  # (output_size)
        da1 = dz2 @ self._w2.T                 # (samples, hidden_size)
        dz1 = da1 * ReLU.gradient(z1)          # (samples, hidden_size)
        dw1 = X.T @ dz1 / n_samples            # (input_size, hidden_size)
        db1 = np.sum(dz1, axis=0) /n_samples   # (hidden_size)

        return dw1, db1, dw2, db2
    
    def predict_proba(self, X):
        _, _, _, a2 = self._forward(X)
        return a2
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities > 0.5,
                        self.classes_[1],
                        self.classes_[0])