import numpy as np

class LogisticRegression:
    def __init__(self, lr=1e-3, n_iterations=100):
        self.lr = lr
        self.n_iterations = n_iterations

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        if np.unique(y).shape[0] != 2:
            raise ValueError("y must be binary")
        
        self.classes_ = np.unique(y)
        y = np.where(y == self.classes_[0], 0, 1).astype(np.float64)

        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        for _ in range(self.n_iterations):
            z = X @ weights + bias  # (n_samples,)
            p = self._sigmoid(z)  # (n_samples,)

            error = p - y

            dw = (1 / n_samples) * (X.T @ error)  # (n_features,)
            db = (1 / n_samples) * np.sum(error)

            weights -= self.lr * dw
            bias -= self.lr * db

        self.weights = weights
        self.bias = bias

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 
                        self.classes_[1],
                        self.classes_[0])