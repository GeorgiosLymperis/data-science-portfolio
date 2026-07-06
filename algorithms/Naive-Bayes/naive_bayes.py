import numpy as np

EPS = 1e-8

class GaussianNB:

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_priors_ = dict(zip(self.classes_, counts / np.sum(counts)))
        self.feature_means_ = {
            c: np.mean(X[y==c, :], axis=0) for c in self.classes_
        }
        self.feature_vars_ = {
            c: np.var(X[y==c, :], axis=0) for c in self.classes_
        }
        self._var_smoothing = 0.01 * max(
            v.max() for v in self.feature_vars_.values()
        )

    def _gaussian_log_likehood(self, X, mean, var):
        return (-0.5 * np.log(2*np.pi*var + self._var_smoothing) 
                - (X - mean)**2 / (2*var + self._var_smoothing))
    
    def _joint_log_likelihood(self, X):
        log_probs_ = {}
        for c in self.classes_:
            log_prob = np.log(self.class_priors_[c])
            log_prob = log_prob + np.sum(
                self._gaussian_log_likehood(X, self.feature_means_[c], self.feature_vars_[c]),
                axis=1
            )
            log_probs_[c] = log_prob  # shape (n_samples,), one log-prob per sample
        return log_probs_

    def predict(self, X):
        log_probs = self._joint_log_likelihood(X)
        log_prob_matrix = np.column_stack([log_probs[c] for c in self.classes_])
        return self.classes_[np.argmax(log_prob_matrix, axis=1)]


class MultinomialNB:
    def __init__(self, *, alpha=1.0): 
        self.alpha = float(alpha)

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        n_features = X.shape[1]
        self.class_priors_ = dict(zip(self.classes_, counts / np.sum(counts)))
        self.feature_log_probs_ = {}

        for c in self.classes_:
            feature_counts = np.sum(X[y==c, :], axis=0)
            total_counts = np.sum(feature_counts)
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = total_counts + self.alpha * n_features
            self.feature_log_probs_[c] = np.log(smoothed_counts / smoothed_total)

    def _joint_log_likelihood(self, x):
        log_probs_ = {}
        for c in self.classes_:
            log_prob = np.log(self.class_priors_[c])
            log_prob = log_prob + np.sum(x * self.feature_log_probs_[c], axis=1)
            log_probs_[c] = log_prob  # shape (n_samples,), one log-prob per sample
        return log_probs_
    
    def predict(self, X):
        log_probs = self._joint_log_likelihood(X)
        log_prob_matrix = np.column_stack([log_probs[c] for c in self.classes_])
        return self.classes_[np.argmax(log_prob_matrix, axis=1)]

class BernoulliNB:
    def __init__(self, *, alpha=1.0): 
        self.alpha = float(alpha)

    @staticmethod
    def _is_binary(X):
        for feature in range(X.shape[1]):
            if np.unique(X[:, feature]).shape[0] > 2:
                return False
        return True
    
    def _binarize(self, X):
        return (X > self._thresholds_).astype(np.float64)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        if not self._is_binary(X):
            raise ValueError("X must be binary")
        self._thresholds_ = X.mean(axis=0)
        X = self._binarize(X)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_priors_ = dict(zip(self.classes_, counts / np.sum(counts)))
        self.feature_probs_ = {}

        for c in self.classes_:
            n_c = np.sum(y == c)
            feature_counts = np.sum(X[y==c, :], axis=0)
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = n_c + self.alpha * 2
            self.feature_probs_[c] = smoothed_counts / smoothed_total


    def _joint_log_likelihood(self, X):
        log_probs = {}
        for c in self.classes_:
            p = self.feature_probs_[c]
            log_prob_present = np.log(p + EPS)
            log_prob_absent = np.log(1 - p + EPS)
            log_prob = np.log(self.class_priors_[c])
            log_prob = log_prob + np.sum(X * log_prob_present + (1 - X) * log_prob_absent, axis=1)
            log_probs[c] = log_prob  # shape (n_samples,), one log-prob per sample
        return log_probs
    
    def predict(self, X):
        X = self._binarize(np.asarray(X, dtype=np.float64))
        log_probs = self._joint_log_likelihood(X)
        log_prob_matrix = np.column_stack([log_probs[c] for c in self.classes_])
        return self.classes_[np.argmax(log_prob_matrix, axis=1)]