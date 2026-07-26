import numpy as np
from sklearn.linear_model import ElasticNet

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, method='eigen'):
        if method == 'eigen':
            self._fit_eigen(X)
        elif method == 'svd':
            self._fit_svd(X)
        else:
            raise NotImplementedError(
                f"Method {method} is not implemented"
            )

        return self

    def _fit_eigen(self, X):
        # X (n, p)
        n, p = X.shape
        if self.n_components > p:
            self.n_components = p
        self.mean = np.mean(X, axis=0)  # (p,)
        X_centered = X - self.mean
        cov = X_centered.T @ X_centered / (n - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

        self.components = eigenvectors[:, :self.n_components]
        self.cumulative_explained_variance = np.sum(self.explained_variance_ratio[:self.n_components])


    def _fit_svd(self, X):
        n, p = X.shape
        if self.n_components > p:
            self.n_components = p
        self.mean = np.mean(X, axis=0)  # (p,)
        X_centered = X - self.mean

        _, S, Vt = np.linalg.svd(X_centered)
        var_eigenvalues = S ** 2 / (n - 1)
        self.explained_variance_ratio = var_eigenvalues / np.sum(var_eigenvalues)

        self.components = Vt.T[:, :self.n_components]
        self.cumulative_explained_variance = np.sum(self.explained_variance_ratio[:self.n_components])


    def transform(self, X):
        X = X - self.mean
        return X @ self.components

class SparsePCA:
    def __init__(self, n_components, alpha, l1_ratio=0.5, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, p = X.shape
        self.mean = np.mean(X, axis=0)  # (p,)
        X_centered = X - self.mean
        A = PCA(self.n_components).fit(X_centered, method='eigen').components
        B = np.zeros((p, self.n_components))

        for _ in range(self.max_iter):
            temp = B.copy()
            for i in range(self.n_components):
                net = ElasticNet(self.alpha, l1_ratio=self.l1_ratio)
                net.fit(X_centered, X_centered @ A[:, i])
                B[:, i] = net.coef_

            U, _, Vt = np.linalg.svd(X_centered.T @ X_centered @ B, full_matrices=False)
            A = U @ Vt

            if np.linalg.norm(temp - B, ord='fro') < self.tol:
                break

        self.components = B / np.linalg.norm(B, axis=0)

    def transform(self, X):
        X = X - self.mean
        return X @ self.components

class RBFKernel:
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

class KernelPCA:
    def __init__(self, n_components, gamma=1):
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = RBFKernel(gamma=gamma)

    def fit(self, X):
        n, p = X.shape
        self.X_train = X
        K = self.kernel(X, X)

        full_n = np.full((n, n), 1/n)
        self.K_ = K - full_n @ K - K @ full_n + full_n @ K @ full_n
        self.kernel_row_means = np.mean(K, axis=1)
        self.kernel_mean = np.mean(self.kernel_row_means)

        eigenvalues, eigenvectors = np.linalg.eigh(self.K_)
        valid_eig = eigenvalues > 1e-10
        eigenvalues = eigenvalues[valid_eig]
        eigenvectors = eigenvectors[:, valid_eig]

        sorted_idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # eigenvectors_norm = eigenvectors / np.sqrt(eigenvalues * n) sklearn follows other normalization
        eigenvectors_norm = eigenvectors / np.sqrt(eigenvalues)
        self.alphas = eigenvectors_norm[:, :self.n_components]
        self.lambdas = eigenvalues[:self.n_components]

        return self

    def transform(self, X):
        K_new = self.kernel(X, self.X_train)  # (n_new, n_train)
        K_new_centered = K_new - K_new.mean(axis=1, keepdims=True) - self.kernel_row_means + self.kernel_mean
        return K_new_centered @ self.alphas