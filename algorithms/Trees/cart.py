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


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth) -> Node:
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

        feature_index, thres, left_ind, right_ind = self._best_split(X, y)

        if feature_index is None:
            node.is_leaf = True
            values, counts = np.unique(y, return_counts=True)
            node.prediction = values[np.argmax(counts)]
            return node
        
        node.feature_index = feature_index
        node.threshold = thres
        node.left = self._build_tree(X[left_ind, :], y[left_ind], depth + 1)
        node.right = self._build_tree(X[right_ind, :], y[right_ind], depth + 1)

        return node
        
    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_thres = None
        best_left_ind = None
        best_right_ind = None

        for feature_index in range(X.shape[1]):
            for thres in np.unique(X[:, feature_index]):
                left_y, right_y, left_ind, right_ind = self._split_data(X, y, feature_index, thres)
                left_y_len = left_y.shape[0]
                right_y_len = right_y.shape[0]
                if left_y_len == 0 or right_y_len == 0:
                    continue
                weighted_gini = (left_y_len * self._gini(left_y) +
                                 right_y_len * self._gini(right_y)) / (
                                     left_y_len + right_y_len
                                 )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_thres = thres
                    best_left_ind = left_ind
                    best_right_ind = right_ind

        return best_feature, best_thres, best_left_ind, best_right_ind
        

    def _gini(self, y):
        p = np.unique_counts(y).counts / y.shape[0]
        return 1 - np.sum(p**2)

    def _split_data(self, X, y, feature_index, threshold):
        mask = X[:, feature_index] <= threshold
        return y[mask], y[~mask], mask, ~mask

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

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, min_variance=0.1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_variance = min_variance

    def fit(self, X, y):
        self.min_variance *= self._variance(y)
        self.root = self._build_tree(X, y, depth=0)

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

        feature_index, thres, left_ind, right_ind = self._best_split(X, y)

        if feature_index is None:
            node.is_leaf = True
            node.prediction = np.mean(y)
            return node
        
        node.feature_index = feature_index
        node.threshold = thres
        node.left = self._build_tree(X[left_ind, :], y[left_ind], depth + 1)
        node.right = self._build_tree(X[right_ind, :], y[right_ind], depth + 1)

        return node
        
    def _best_split(self, X, y):
        best_variance = float('inf')
        best_feature = None
        best_thres = None
        best_left_ind = None
        best_right_ind = None

        for feature_index in range(X.shape[1]):
            for thres in np.unique(X[:, feature_index]):
                left_y, right_y, left_ind, right_ind = self._split_data(X, y, feature_index, thres)
                left_y_len = left_y.shape[0]
                right_y_len = right_y.shape[0]
                if left_y_len == 0 or right_y_len == 0:
                    continue
                weighted_variance = (left_y_len * self._variance(left_y) + 
                                    right_y_len * self._variance(right_y)) / (
                                        left_y_len + right_y_len
                                    )

                if weighted_variance < best_variance:
                    best_variance = weighted_variance
                    best_feature = feature_index
                    best_thres = thres
                    best_left_ind = left_ind
                    best_right_ind = right_ind

        return best_feature, best_thres, best_left_ind, best_right_ind
        

    def _variance(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _split_data(self, X, y, feature_index, threshold):
        mask = X[:, feature_index] <= threshold
        return y[mask], y[~mask], mask, ~mask

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