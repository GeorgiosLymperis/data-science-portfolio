import numpy as np
from cart import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, lr=0.1, max_depth=2,
                 min_samples_split=2, max_features=5, subsample=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state


    def fit(self, X, y):
        n_samples = X.shape[0]
        subsample_size = max(1, round(self.subsample * n_samples))
        self._initial_prediction = np.mean(y)
        current_prediction = np.full(n_samples, self._initial_prediction)

        rng = np.random.default_rng(self.random_state)
        tree_seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

        self.trees = [DecisionTreeRegressor(self.max_depth,
                                            self.min_samples_split,
                                            max_features=self.max_features,
                                            random_state=seed)
                      for seed in tree_seeds]

        for tree in self.trees:
            residuals = y - current_prediction
            sample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
            tree.fit(X[sample_idx], residuals[sample_idx])
            tree_prediction = tree.predict(X)
            current_prediction += self.lr * tree_prediction

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self._initial_prediction)

        for tree in self.trees:
            tree_prediction = tree.predict(X)
            predictions += self.lr * tree_prediction

        return predictions
        

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, lr=0.1, max_depth=2,
                 min_samples_split=2, max_features=5, subsample=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.max_features = max_features
        self.subsample = subsample

    @staticmethod
    def _sigmoid(score):
        return 1 / (1 + np.exp(-score))

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        if self.classes_.shape[0] > 2:
            raise NotImplementedError("Only binary classification is supported")
        n_samples = X.shape[0]
        subsample_size = max(1, round(self.subsample * n_samples))
        base_rate = np.mean(y)
        self._initial_raw_score = np.log(base_rate / (1 - base_rate))
        current_raw_score = np.full(n_samples, self._initial_raw_score)

        rng = np.random.default_rng(self.random_state)
        tree_seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

        self.trees = [DecisionTreeRegressor(self.max_depth,
                                            self.min_samples_split,
                                            max_features=self.max_features,
                                            random_state=seed)
                      for seed in tree_seeds]

        for tree in self.trees:
            residuals = y - self._sigmoid(current_raw_score)
            sample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
            tree.fit(X[sample_idx], residuals[sample_idx])
            tree_prediction = tree.predict(X)
            current_raw_score += self.lr * tree_prediction

    def predict_proba(self, X):
        n_samples = X.shape[0]
        raw_score = np.full(n_samples, self._initial_raw_score)
        for tree in self.trees:
            raw_score += self.lr * tree.predict(X)

        return self._sigmoid(raw_score)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.where(probs >= 0.5, 1, 0)]