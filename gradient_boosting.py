import numpy as np
from typing import Tuple
from decision_tree import DecisionTreeRegressor
from scipy.special import expit
from dataclasses import dataclass

@dataclass
class Node:
    """
    A class representing a node in a decision tree.

    Attributes:
        feature (int): The feature index used for splitting the node.
        threshold (float): The threshold value used for the split.
        n_samples (int): The number of samples in the node.
        value (float): The predicted value of the node.
        mse (float): The mean squared error of the node.
        left (Node): The left child node.
        right (Node): The right child node.
    """
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: float = None
    mse: float = None
    left: 'Node' = None
    right: 'Node' = None

class GradientBoostingClassifier:
    """
    A Gradient Boosting classifier for binary classification tasks.

    This implementation uses decision trees as the base estimators and log-loss as the default loss function.

    Attributes:
        n_estimators (int): The number of boosting iterations.
        learning_rate (float): The step size applied to each tree's contribution.
        max_depth (int): The maximum depth of the individual decision trees.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        loss (str or callable): The loss function to be optimized. Defaults to "logloss".
        verbose (bool): Whether to print progress during training.
        base_pred_ (float): The initial prediction (log-odds) for all samples.
        trees_ (list): List of fitted decision trees.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="logloss",
        verbose=False,
    ):
        """
        Initialize the GradientBoostingClassifier.

        Args:
            n_estimators (int): The number of boosting stages to run. Default is 100.
            learning_rate (float): The contribution of each tree. Default is 0.1.
            max_depth (int): The maximum depth of the trees. Default is 3.
            min_samples_split (int): The minimum number of samples required to split a node.
            loss (str or callable): The loss function to be used. Default is "logloss".
            verbose (bool): If True, prints progress during training. Default is False.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.base_pred_ = None
        self.trees_ = []

    def _logloss(self, y_true, y_pred_proba) -> Tuple[float, np.ndarray]:
        """
        Compute the log-loss and its gradient.

        Args:
            y_true (np.ndarray): True binary labels.
            y_pred_proba (np.ndarray): Predicted probabilities.

        Returns:
            Tuple[float, np.ndarray]: The log-loss value and the gradient (residuals).
        """
        loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        residuals = y_true - y_pred_proba
        return loss, residuals

    def _calculate_gamma(self, y_true, y_pred_proba) -> float:
        """
        Calculate the optimal update step (gamma) for a leaf node.

        Args:
            y_true (np.ndarray): True binary labels for the samples in the leaf.
            y_pred_proba (np.ndarray): Predicted probabilities for the samples in the leaf.

        Returns:
            float: The optimal gamma value for the leaf.
        """
        numerator = np.sum(y_true - y_pred_proba)
        denominator = np.sum(y_pred_proba * (1 - y_pred_proba))
        if denominator == 0:
            return 0
        return numerator / denominator

    def fit(self, X, y):
        """
        Fit the GradientBoostingClassifier model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target binary labels of shape (n_samples,).
        """
        mean_pred = np.mean(y)
        self.base_pred_ = np.log(mean_pred / (1 - mean_pred))

        y_pred_log_odds = np.full(len(y), self.base_pred_)

        for iteration in range(self.n_estimators):
            y_pred_proba = expit(y_pred_log_odds)  # sigmoid function
            if self.loss == "logloss":
                loss, residuals = self._logloss(y, y_pred_proba)
            else:
                loss, residuals = self.loss(y, y_pred_proba)

            if self.verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}: Loss = {loss}")

            tree = DecisionTreeRegressor(max_depth=self.max_depth, \
                min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)

            # Update predictions with calculated gamma
            leaf_nodes = self._get_leaf_nodes(tree.tree_)
            for leaf in leaf_nodes:
                indices = self._get_samples_in_leaf(leaf, tree.tree_, X)
                gamma = self._calculate_gamma(y[indices], expit(y_pred_log_odds[indices]))
                y_pred_log_odds[indices] += self.learning_rate * gamma

            self.trees_.append(tree)

    def _get_leaf_nodes(self, node: Node) -> list:
        """
        Retrieve all leaf nodes from a decision tree.

        Args:
            node (Node): The root node of the tree.

        Returns:
            list: A list of leaf nodes.
        """
        if node.left is None and node.right is None:
            return [node]
        leaves = []
        if node.left:
            leaves.extend(self._get_leaf_nodes(node.left))
        if node.right:
            leaves.extend(self._get_leaf_nodes(node.right))
        return leaves

    def _get_samples_in_leaf(self, leaf: Node, tree: Node, X: np.ndarray) -> np.ndarray:
        """
        Get the indices of samples that fall into a specific leaf node.

        Args:
            leaf (Node): The target leaf node.
            tree (Node): The root node of the tree.
            X (np.ndarray): The feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of indices of samples that fall into the leaf node.
        """
        indices = []
        for i, x in enumerate(X):
            node = tree
            while node.left is not None and node.right is not None:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            if node == leaf:
                indices.append(i)
        return np.array(indices)

    def predict(self, X) -> np.ndarray:
        """
        Predict the binary labels for the input samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted binary labels of shape (n_samples,).
        """
        predictions_log_odds = np.full(len(X), self.base_pred_)
        for tree in self.trees_:
            predictions_log_odds += self.learning_rate * tree.predict(X)
        proba = expit(predictions_log_odds)
        return (proba >= 0.5).astype(int)
