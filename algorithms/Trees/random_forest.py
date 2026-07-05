from cart import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from scipy import stats

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2,
                 max_features=5, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        tree_seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

        self.trees = [DecisionTreeClassifier(self.max_depth, self.min_samples_split,
                                              self.max_features, random_state=seed)
                      for seed in tree_seeds]

        oob_vote_counts = np.zeros((X.shape[0], np.unique(y).shape[0]), dtype=np.int64)
        for tree in self.trees:
            X_sample, y_sample, oob_mask = self._bootstrap_sample(X, y, rng)
            tree.fit(X_sample, y_sample)
            oob_pred = tree.predict(X[oob_mask])
            oob_vote_counts[oob_mask, oob_pred] += 1

        self.oob_score = self._compute_oob_score(oob_vote_counts, y)

    def _compute_oob_score(self, oob_vote_counts, y):
        has_votes = oob_vote_counts.sum(axis=1) > 0
        predicted_classes = np.argmax(oob_vote_counts[has_votes], axis=1)
        return np.mean(y[has_votes] == predicted_classes)

    def _bootstrap_sample(self, X, y, rng):
        n = X.shape[0]
        indices = rng.choice(n, n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[indices] = False
        return X[indices, :], y[indices], oob_mask
    
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=np.int64)
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)

        return stats.mode(predictions, axis=0)[0]


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2,
                 min_variance=0.1, max_features=5, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_variance = min_variance
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        tree_seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

        self.trees = [DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.min_variance,
                                             self.max_features, random_state=seed)
                      for seed in tree_seeds]

        oob_sum = np.zeros(X.shape[0], dtype=np.float64)
        oob_count = np.zeros(X.shape[0], dtype=np.int64)
        for tree in self.trees:
            X_sample, y_sample, oob_mask = self._bootstrap_sample(X, y, rng)
            tree.fit(X_sample, y_sample)
            oob_sum[oob_mask] += tree.predict(X[oob_mask])
            oob_count[oob_mask] += 1

        self.oob_score = self._compute_oob_score(oob_sum, oob_count, y)

    def _compute_oob_score(self, oob_sum, oob_count, y):
        has_votes = oob_count > 0
        oob_preds = oob_sum[has_votes] / oob_count[has_votes]
        y_has = y[has_votes]
        ss_res = np.sum((y_has - oob_preds) ** 2)
        ss_tot = np.sum((y_has - np.mean(y_has)) ** 2)
        return 1 - ss_res / ss_tot

    def _bootstrap_sample(self, X, y, rng):
        n = X.shape[0]
        indices = rng.choice(n, n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[indices] = False
        return X[indices, :], y[indices], oob_mask

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=np.float64)
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)

        return predictions.mean(axis=0)
