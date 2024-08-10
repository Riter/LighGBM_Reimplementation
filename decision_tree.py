from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: float = None
    mse: float = None
    left: Node = None
    right: Node = None

@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2
    n_bins: int = 10  # Number of bins for histogram-based splitting

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._fit_leaf_wise(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        mse_error = np.mean((y - np.mean(y)) ** 2)
        return mse_error

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mse criterion for two given sets of target values."""
        mse_left, mse_right = self._mse(y_left), self._mse(y_right)
        n_left, n_right = y_left.shape[0], y_right.shape[0]
        return (n_left * mse_left + n_right * mse_right) / (n_left + n_right)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node using histogram-based algorithm."""
        node_size = y.size
        n_features_ = X.shape[1]
        if node_size < self.min_samples_split:
            return None, None

        node_mse = self._mse(y)
        best_mse = node_mse
        best_idx, best_thr = None, None

        for idx in range(n_features_):
            # Create histogram for current feature
            hist, bin_edges = np.histogram(X[:, idx], bins=self.n_bins)

            # Calculate MSE for each bin split
            for i in range(1, len(bin_edges)):
                thr = bin_edges[i]
                left_mask = X[:, idx] <= thr
                right_mask = X[:, idx] > thr

                left, right = y[left_mask], y[right_mask]

                if left.size == 0 or right.size == 0:
                    continue

                weighted_mse = self._weighted_mse(left, right)
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def _fit_leaf_wise(self, X: np.ndarray, y: np.ndarray) -> Node:
        """Build a decision tree in a leaf-wise manner."""
        # Initialize the root node
        root = Node(
            n_samples=X.shape[0],
            value=np.mean(y),
            mse=self._mse(y)
        )

        # Queue for nodes to split
        queue = deque([(root, X, y, 0)])

        while queue:
            node, X_node, y_node, depth = queue.popleft()

            if depth >= self.max_depth or X_node.shape[0] < self.min_samples_split:
                continue

            best_idx, best_thr = self._best_split(X_node, y_node)
            if best_idx is None or best_thr is None:
                continue

            # Create masks for splitting
            left_mask = X_node[:, best_idx] <= best_thr
            right_mask = X_node[:, best_idx] > best_thr

            # Create child nodes
            X_left, y_left = X_node[left_mask], y_node[left_mask]
            X_right, y_right = X_node[right_mask], y_node[right_mask]

            node.feature = best_idx
            node.threshold = best_thr

            node.left = Node(
                n_samples=X_left.shape[0],
                value=np.mean(y_left),
                mse=self._mse(y_left)
            )
            node.right = Node(
                n_samples=X_right.shape[0],
                value=np.mean(y_right),
                mse=self._mse(y_right)
            )

            # Add child nodes to the queue
            queue.append((node.left, X_left, y_left, depth + 1))
            queue.append((node.right, X_right, y_right, depth + 1))

        return root

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        predictions = []
        for obj in X:
            prediction = self._predict_one_sample(obj)
            predictions.append(prediction)

        return np.array(predictions)

    def _predict_one_sample(self, features: np.ndarray) -> float:
        """Predict the target value of a single sample."""
        node = self.tree_
        while node.left is not None and node.right is not None:
            if features[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
