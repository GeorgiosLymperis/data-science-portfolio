import numpy as np
from dataclasses import dataclass

@dataclass
class Node:
    n_samples: int
    is_leaf: bool = False
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class IsolationTree:
    def __init__(self, height_limit, seed=42):
        self.height_limit = height_limit
        self.root = None
        self.rng_ = np.random.default_rng(seed=seed)

    def fit(self, X):
        self.root = self._build_tree(X, depth=0)
        return self

    def _build_tree(self, X, depth):
        n_samples, p = X.shape
        node = Node(n_samples=n_samples, is_leaf=False)
        if (depth >= self.height_limit or
            node.n_samples <= 1):
            node.is_leaf = True
            return node

        feature_index = self.rng_.integers(0, p, size=1)[0]
        threshold = self.rng_.uniform(X[:, feature_index].min(), X[:, feature_index].max())
        mask = X[:, feature_index] <= threshold
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[mask], depth + 1)
        node.right = self._build_tree(X[~mask], depth + 1)

        return node

    def path_length(self, x, node=None, current_height=0):
        if node is None:
            node = self.root
        if node.is_leaf:
            return current_height + c(node.n_samples)
        if x[node.feature_index] <= node.threshold:
            return self.path_length(x, node.left, current_height + 1)
        else:
            return self.path_length(x, node.right, current_height + 1)
        

class IsolationForest:
    def __init__(self, n_trees=100, subsample_size=256, contamination=0.1, seed=42):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.contamination = contamination
        self.threshold = None
        self.trees = []
        self.rng_ = np.random.default_rng(seed=seed)

    def fit(self, X):
        n = X.shape[0]
        self.psi = min(self.subsample_size, n)
        height_limit = int(np.ceil(np.log2(self.psi)))
        for _ in range(self.n_trees):
            sample = self.rng_.integers(0, n, size=self.psi)
            tree = IsolationTree(height_limit, seed=self.rng_.integers(0, 2**32 - 1, size=1)[0]).fit(X[sample])
            self.trees.append(tree)

        train_scores = self.anomaly_score(X)
        self.threshold = np.quantile(train_scores, 1 - self.contamination)

        return self

    def anomaly_score(self, X):
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            avg_path = np.mean([tree.path_length(X[i]) for tree in self.trees])
            scores[i] = 2 ** (-avg_path / c(self.psi))
        return scores

    def predict(self, X):
        scores = self.anomaly_score(X)
        return np.where(scores > self.threshold, -1, 1)

    def decision_function(self, X):
        return self.threshold - self.anomaly_score(X)

def c(n):
    if n <= 1:
        return 0
    if n == 2:
        return 1
    return 2*H(n-1) - 2*(n-1)/n

def H(i):
    return np.log(i) + np.euler_gamma