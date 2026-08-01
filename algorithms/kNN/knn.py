import numpy as np
from scipy.stats import mode

class kNNRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', p=2, standarize=False):
        self.n_neighbors_ = n_neighbors
        self.weights_ = weights
        self.p_ = p
        self.standarize_ = standarize

    def fit(self, X, y):
        if self.standarize_:
            self.mean_ = np.mean(X, axis=0)  # (p,)
            self.std_ = np.std(X, axis=0)  # (p,)
            self.X_train_ = (X - self.mean_) / self.std_
            self.y = y

            return self

        self.X_train_ = X
        self.y = y
        return self

    def distances_(self, X):
        if self.standarize_:
            X = (X - self.mean_) / self.std_

        distances = np.zeros((X.shape[0], self.X_train_.shape[0]))  # (predictions, n)
        for i in range(X.shape[0]):
            for j in range(self.X_train_.shape[0]):
                x0 = X[i, :]
                x1 = self.X_train_[j, :]
                distances[i, j] = np.sum(np.abs(x0 - x1) ** self.p_) ** (1 / self.p_)

        return distances
         

    def predict(self, X):
        distances = self.distances_(X)
        best_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors_]
        if self.weights_ == 'uniform':
            return np.mean(self.y[best_idx], axis=1)
        elif self.weights_ == 'distance':
            d = np.take_along_axis(distances, best_idx, axis=1)
            d = np.where(d == 0, 1e-12, d)
            weight = 1 / d
            return np.sum(weight * self.y[best_idx], axis=1) / np.sum(weight, axis=1)


class kNNClassifier:
    def __init__(self, n_neighbors=5, p=2, standarize=False):
        self.n_neighbors_ = n_neighbors
        self.p_ = p
        self.standarize_ = standarize

    
    def fit(self, X, y):
        if self.standarize_:
            self.mean_ = np.mean(X, axis=0)  # (p,)
            self.std_ = np.std(X, axis=0)  # (p,)
            self.X_train_ = (X - self.mean_) / self.std_
            self.y = y

            return self

        self.X_train_ = X
        self.y = y
        return self

    def distances_(self, X):
        if self.standarize_:
            X = (X - self.mean_) / self.std_

        distances = np.zeros((X.shape[0], self.X_train_.shape[0]))  # (predictions, n)
        for i in range(X.shape[0]):
            for j in range(self.X_train_.shape[0]):
                x0 = X[i, :]
                x1 = self.X_train_[j, :]
                distances[i, j] = np.sum(np.abs(x0 - x1) ** self.p_) ** (1 / self.p_)

        return distances

    def predict(self, X):
        distances = self.distances_(X)
        best_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors_]
        return mode(self.y[best_idx], axis=1, keepdims=False).mode