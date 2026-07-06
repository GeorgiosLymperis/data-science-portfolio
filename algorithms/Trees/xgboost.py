import numpy as np
from dataclasses import dataclass

@dataclass
class Node:
    n_samples: int
    impurity: float
    is_leaf: bool = False
    prediction: int | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2,
                 max_features=None, random_state=None,
                 reg_lambda=1e-2, gamma=0.0):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.gamma = gamma

    def fit(self, X, gradients, hessians):
        n_features = X.shape[1]
        self.n_features_to_sample = self._get_max_features_to_sample(n_features)
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._build_tree(X, 0, gradients, hessians)

    def _get_max_features_to_sample(self, n_features):
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float) and 0 <= self.max_features <= 1.0:
            return max(1, round(self.max_features * n_features))
        else:
            raise ValueError("max_features should be None or int or float between 0 and 1. Got", self.max_features)
        
    def _score(self, gradients, hessians):
        G, H = gradients.sum(), hessians.sum()
        return -0.5 * G**2 / (H + self.reg_lambda)

    def _build_tree(self, X, depth, gradients, hessians) -> Node:
        n_samples = X.shape[0]
        impurity = self._score(gradients, hessians)
        node = Node(n_samples=n_samples, impurity=impurity, is_leaf=False)


        if (depth >= self.max_depth or
            node.n_samples < self.min_samples_split):
            node.is_leaf = True
            G, H = gradients.sum(), hessians.sum()
            node.prediction = -G / (H + self.reg_lambda)

            return node

        feature_index, thres, left_ind, right_ind = self._best_split(X, gradients, hessians)

        if feature_index is None:
            node.is_leaf = True
            G, H = gradients.sum(), hessians.sum()
            node.prediction = -G / (H + self.reg_lambda)
            return node

        node.feature_index = feature_index
        node.threshold = thres
        node.left = self._build_tree(X[left_ind, :], depth + 1, gradients[left_ind], hessians[left_ind])
        node.right = self._build_tree(X[right_ind, :], depth + 1, gradients[right_ind], hessians[right_ind])

        return node

    def _best_split(self, X, gradients, hessians):
        best_gain = 0.0
        best_feature = None
        best_thres = None
        best_left_ind = None
        best_right_ind = None

        G = gradients.sum()
        H = hessians.sum()

        candidate_features = self._rng.choice(X.shape[1], size=self.n_features_to_sample, replace=False)

        for feature_index in candidate_features:
            for thres in np.unique(X[:, feature_index]):
                left_ind, right_ind = self._split_data(X, feature_index, thres)
                if left_ind.sum() == 0 or right_ind.sum() == 0:
                    continue

                G_l, H_l = gradients[left_ind].sum(), hessians[left_ind].sum()
                G_r, H_r = gradients[right_ind].sum(), hessians[right_ind].sum()

                gain = 0.5 * (G_l**2/(H_l+self.reg_lambda) + G_r**2/(H_r+self.reg_lambda)
                              - G**2/(H+self.reg_lambda)) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_thres = thres
                    best_left_ind = left_ind
                    best_right_ind = right_ind

        return best_feature, best_thres, best_left_ind, best_right_ind

    def _split_data(self, X, feature_index, threshold):
        mask = X[:, feature_index] <= threshold
        return mask, ~mask

    def _traverse(self, X, node: Node):
        if node.is_leaf:
            return node.prediction
        
        if X[node.feature_index] <= node.threshold:
            return self._traverse(X, node.left)
        else:
            return self._traverse(X, node.right)

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.float64)
        for sample in range(n_samples):
            predictions[sample] = self._traverse(X[sample, :], self.root)

        return predictions
    
class SquaredLoss:
    def gradient(self, y, pred):
        return pred - y
    
    def hessian(self, y, pred):
        return np.ones_like(y, dtype=np.float64)
    
class LogisticLoss:
    def gradient(self, y, p):
        return p - y
    
    def hessian(self, y, p):
        return p * (1 - p)
    
class XGBoostRegressor:
    def __init__(self, n_estimators=100, lr=0.1, max_depth=2,
                 min_samples_split=2, max_features=5, subsample=1.0,
                reg_lambda=1e-2, gamma=0.0, random_state=42):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.gamma = gamma


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
                                            random_state=seed,
                                            reg_lambda=self.reg_lambda,
                                            gamma=self.gamma)
                      for seed in tree_seeds]

        loss = SquaredLoss()
        for tree in self.trees:
            gradients = loss.gradient(y, current_prediction)
            hessians = loss.hessian(y, current_prediction)
            sample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
            tree.fit(X[sample_idx], gradients[sample_idx], hessians[sample_idx])
            tree_prediction = tree.predict(X)
            current_prediction += self.lr * tree_prediction

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self._initial_prediction)

        for tree in self.trees:
            tree_prediction = tree.predict(X)
            predictions += self.lr * tree_prediction

        return predictions
    

class XGBoostClassifier:
    def __init__(self, n_estimators=100, lr=0.1, max_depth=2,
                 min_samples_split=2, max_features=5, subsample=1.0,
                reg_lambda=1e-2, gamma=0.0, random_state=42):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.max_features = max_features
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma

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
                                            random_state=seed,
                                            reg_lambda=self.reg_lambda,
                                            gamma=self.gamma)
                      for seed in tree_seeds]

        loss = LogisticLoss()
        for tree in self.trees:
            p = self._sigmoid(current_raw_score)
            gradients = loss.gradient(y, p)
            hessians = loss.hessian(y, p)
            sample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
            tree.fit(X[sample_idx], gradients[sample_idx], hessians[sample_idx])
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