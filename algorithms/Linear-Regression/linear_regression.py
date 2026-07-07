import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            beta = beta - lr * gradient
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def ols(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self
    
class RidgeRegression:
    def __init__(self, labmda: float = 1.0):
        self.coef_ = None
        self.intercept_ = None
        self.lambda_ = labmda

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            gradient[1:] += 2 * self.lambda_ * beta[1:] / n  # L2 regularization
            beta = beta - lr * gradient
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def ols(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        penalty = self.lambda_ * np.eye(X.shape[1])
        penalty[0, 0] = 0  # do not regularize the intercept
        beta = np.linalg.inv(X.T @ X + penalty) @ X.T @ y
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self


class LassoRegression:
    def __init__(self, labmda: float = 1.0):
        self.coef_ = None
        self.intercept_ = None
        self.lambda_ = labmda

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            beta_temp = beta - lr * gradient
            threshold = lr * self.lambda_ / n
            beta[1:] = np.sign(beta_temp[1:]) * np.maximum(np.abs(beta_temp[1:]) - threshold, 0)
            beta[0] = beta_temp[0]
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    

class ElasticNet:
    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        self.coef_ = None
        self.intercept_ = None
        self.l1 = l1
        self.l2 = l2

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            gradient[1:] += 2 * self.l2 * beta[1:] / n  # L2 regularization
            beta_temp = beta - lr * gradient
            threshold = lr * self.l1 / n
            beta[1:] = np.sign(beta_temp[1:]) * np.maximum(np.abs(beta_temp[1:]) - threshold, 0)
            beta[0] = beta_temp[0]
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_