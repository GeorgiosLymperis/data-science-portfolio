import numpy as np
from scipy.linalg import solve_triangular

class GaussianMixture:
    def __init__(self, n_components=1, tol=0.001, max_iter=100, reg_covar=1e-6, random_state=42):
        self.n_components_ = n_components
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.reg_covar_ = reg_covar
        self.rng_ = np.random.default_rng(random_state)
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_ = None
        self.precisions_cholesky_ = None
        self.converged_ = False
        self.log_history_ = []

    def fit(self, X):
        n, p = X.shape
        self.weights_ = np.full(self.n_components_, 1 / self.n_components_)  # (c,)
        init_idx = self.rng_.choice(n, size=self.n_components_, replace=False)
        self.means_ = X[init_idx].copy()  # (c, p)
        self.covariances_ = np.tile(np.cov(X, rowvar=False), (self.n_components_, 1, 1))  # (c, p, p)
        self.precisions_cholesky_ = np.empty((self.n_components_, p, p))
        self.precisions_ = np.empty((self.n_components_, p, p))
        for c in range(self.n_components_):
            self._update_precision(c, p)

        log_likelihood_prev = -float('inf')
        log_prob = np.empty((n, self.n_components_))
        log_weighted = np.empty((n, self.n_components_))
        log_norm = np.empty((n,))

        for iter in range(self.max_iter_):
            for c in range(self.n_components_):
                log_prob[:, c] = self.log_gaussian_pdf(X, self.means_[c, :], self.precisions_cholesky_[c, :])

                log_weighted[:, c] = np.log(self.weights_[c]) + log_prob[:, c]
            log_norm = self.logsumexp(log_weighted, axis=1)

            log_gamma = log_weighted - log_norm[:, None]
            gamma = np.exp(log_gamma)

            N_k = np.sum(gamma, axis=0) + 10 * np.finfo(gamma.dtype).eps
            for c in range(self.n_components_):
                self.means_[c, :] = np.sum(gamma[:, c][:, None] * X, axis=0) / N_k[c]
                diff = X - self.means_[c, :]
                self.covariances_[c, :] = np.sum(gamma[:, c][:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0) / N_k[c]
                self.covariances_[c, :] += self.reg_covar_ * np.identity(p)
                self._update_precision(c, p)

            self.weights_ = N_k / n

            log_likelihood = np.sum(log_norm)
            self.log_history_.append(log_likelihood / n)
            if np.abs(log_likelihood - log_likelihood_prev) / n < self.tol_:
                self.converged_ = True
                break
            log_likelihood_prev = log_likelihood    


        return self


    def _update_precision(self, c, p):
        cov_chol = np.linalg.cholesky(self.covariances_[c, :])
        self.precisions_cholesky_[c, :] = solve_triangular(cov_chol, np.identity(p), lower=True).T
        self.precisions_[c, :] = self.precisions_cholesky_[c, :] @ self.precisions_cholesky_[c, :].T

    def log_gaussian_pdf(self, X, mu, prec_chol):
        p = X.shape[1]
        log_det_chol = np.sum(np.log(np.diag(prec_chol)))  # = -0.5 * log|cov|
        diff = X - mu[None, :]  # (n, p)
        y = diff @ prec_chol  # (n, p)
        mahalanobis = np.sum(y ** 2, axis=1)  # (n,)

        return -0.5 * p * np.log(2 * np.pi) - 0.5 * mahalanobis + log_det_chol

    def logsumexp(self, v, axis):
        m = np.max(v, axis=axis, keepdims=True)
        s = m + np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True))
        return np.squeeze(s, axis=axis)

    def predict(self, X):
        log_prob = np.empty((X.shape[0], self.n_components_))
        for c in range(self.n_components_):
            log_prob[:, c] = self.log_gaussian_pdf(X, self.means_[c, :], self.precisions_cholesky_[c, :])
        log_prob += np.log(self.weights_)
        return np.argmax(log_prob, axis=1)