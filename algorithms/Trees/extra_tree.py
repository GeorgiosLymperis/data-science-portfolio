import numpy as np
from dataclasses import dataclass
from scipy import stats

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

class ExtraTreeClassifier:
    def __init__(self,max_depth=5, min_samples_split=2,
                 max_features: int | str = 5, random_state=None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        if isinstance(self.max_features, int):
            self.max_features = min(self.max_features, X.shape[1])
        elif self.max_features == 'sqrt':
            self.max_features = int(np.floor(np.sqrt(X.shape[1])))
        else:
            raise ValueError("max_features should be int or 'sqrt'. Got", self.max_features)
        self.root = self._build_tree(X, y, depth=0)

        return self

    def _build_tree(self, X, y, depth):
        n_samples = y.shape[0]
        impurity = self._gini(y)
        node = Node(n_samples=n_samples, impurity=impurity, is_leaf=False)


        if (depth >= self.max_depth or 
            node.n_samples < self.min_samples_split or
            np.unique(y).shape[0] == 1):
            node.is_leaf = True
            values, counts = np.unique(y, return_counts=True)
            node.prediction = values[np.argmax(counts)]
            return node

        best_mask = None
        best_thres = None
        best_score = -float('inf')
        best_feature = None
        candidate_features = self.rng.choice(X.shape[1], size=self.max_features, replace=False)

        for feature in candidate_features:
            low, high = np.min(X[:, feature]), np.max(X[:, feature])
            if low == high:
                continue
            thres = self.rng.uniform(low, high)

            mask = X[:, feature] <= thres
            gain = self._gain(impurity, y[mask], y[~mask])
            if gain > best_score:
                best_score = gain
                best_mask = mask
                best_thres = thres
                best_feature = feature

        if best_feature is None:
            node.is_leaf = True
            values, counts = np.unique(y, return_counts=True)
            node.prediction = values[np.argmax(counts)]
            return node

        node.feature_index = best_feature
        node.threshold = best_thres
        node.left = self._build_tree(X[best_mask, :], y[best_mask], depth + 1)
        node.right = self._build_tree(X[~best_mask, :], y[~best_mask], depth + 1)
        return node

    def _gini(self, y):
        p = np.unique_counts(y).counts / y.shape[0]
        return 1 - np.sum(p**2)

    def _gain(self, impurity, y_left, y_right):
        n_y_left, n_y_right = y_left.shape[0], y_right.shape[0]
        left_impurity = self._gini(y_left)
        right_impurity = self._gini(y_right)
        return impurity - (n_y_left * left_impurity + n_y_right * right_impurity) / (n_y_left + n_y_right)

    def _traverse(self, X, node: Node):
        if node.is_leaf:
            return node.prediction
        
        if X[node.feature_index] <= node.threshold:
            return self._traverse(X, node.left)
        else:
            return self._traverse(X, node.right)

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.int64)
        for sample in range(n_samples):
            predictions[sample] = self._traverse(X[sample, :], self.root)

        return predictions

class ExtraTreesClassifier:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2,
                 max_features=5, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        tree_seeds = self.rng.integers(0, 2 ** 32 - 1, size=self.n_estimators)

        self.trees = [ExtraTreeClassifier(self.max_depth, self.min_samples_split,
                                          self.max_features, seed) for seed in tree_seeds]

        for tree in self.trees:
            tree.fit(X, y)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=np.int64)
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)

        return stats.mode(predictions, axis=0)[0]


class ExtraTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, min_variance=0.1,
                 max_features=None, random_state=None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_variance = min_variance
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = X.shape[1]
        elif isinstance(self.max_features, int):
            self.max_features = min(self.max_features, X.shape[1])
        elif self.max_features == 'sqrt':
            self.max_features = int(np.floor(np.sqrt(X.shape[1])))
        else:
            raise ValueError("max_features should be None, int or 'sqrt'. Got", self.max_features)
        self.min_variance *= self._variance(y)
        self.root = self._build_tree(X, y, depth=0)

        return self

    def _build_tree(self, X, y, depth) -> Node:
        n_samples = y.shape[0]
        impurity = self._variance(y)
        node = Node(n_samples=n_samples, impurity=impurity, is_leaf=False)


        if (depth >= self.max_depth or 
            node.n_samples < self.min_samples_split or
            impurity < self.min_variance or impurity < 1e-3):
            node.is_leaf = True
            node.prediction = np.mean(y)
            return node

        best_mask = None
        best_thres = None
        best_score = -float('inf')
        best_feature = None
        candidate_features = self.rng.choice(X.shape[1], size=self.max_features, replace=False)
        
        for feature in candidate_features:
            low, high = np.min(X[:, feature]), np.max(X[:, feature])
            if low == high:
                continue
            thres = self.rng.uniform(low, high)

            mask = X[:, feature] <= thres
            gain = self._gain(impurity, y[mask], y[~mask])
            if gain > best_score:
                best_score = gain
                best_mask = mask
                best_thres = thres
                best_feature = feature

        if best_feature is None:
            node.is_leaf = True
            node.prediction = np.mean(y)
            return node

        node.feature_index = best_feature
        node.threshold = best_thres
        node.left = self._build_tree(X[best_mask, :], y[best_mask], depth + 1)
        node.right = self._build_tree(X[~best_mask, :], y[~best_mask], depth + 1)
        return node
        

    def _variance(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _gain(self, impurity, y_left, y_right):
        n_y_left, n_y_right = y_left.shape[0], y_right.shape[0]
        left_impurity = self._variance(y_left)
        right_impurity = self._variance(y_right)
        return impurity - (n_y_left * left_impurity + n_y_right * right_impurity) / (n_y_left + n_y_right)


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

class ExtraTreesRegressor:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2,
                 min_variance=0.1, max_features=5, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_variance = min_variance
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        tree_seeds = self.rng.integers(0, 2 ** 32 - 1, size=self.n_estimators)

        self.trees = [ExtraTreeRegressor(self.max_depth, self.min_samples_split,
                                          self.min_variance, self.max_features, seed) 
                                          for seed in tree_seeds]

        for tree in self.trees:
            tree.fit(X, y)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=np.float64)
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)

        return np.mean(predictions, axis=0)
