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
