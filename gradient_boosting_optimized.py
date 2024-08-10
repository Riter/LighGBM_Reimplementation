import numpy as np
from typing import Tuple
from decision_tree import DecisionTreeRegressor
from scipy.special import expit
from gradient_boosting import Node

class GradientBoostingClassifierOptimized:
    """
    An optimized Gradient Boosting classifier for binary classification tasks.

    This implementation includes support for Gradient-based One-Side Sampling (GOSS) and 
    Exclusive Feature Bundling (EFB) for improved efficiency.

    Attributes:
        n_estimators (int): The number of boosting iterations.
        learning_rate (float): The step size applied to each tree's contribution.
        max_depth (int): The maximum depth of the individual decision trees.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        loss (str or callable): The loss function to be optimized. Defaults to "logloss".
        verbose (bool): Whether to print progress during training.
        boosting_type (str): The type of boosting algorithm to use ("goss" or "gbdt").
        alpha (float): The fraction of high gradient data used in GOSS.
        beta (float): The fraction of low gradient data sampled in GOSS.
        apply_efb (bool): Whether to apply Exclusive Feature Bundling (EFB) to the data.
        K (int): Conflict threshold for EFB bundling.
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
        boosting_type="goss",
        alpha=0.2,
        beta=0.1,
        apply_efb=False,
        K=1              # Conflict threshold for EFB
    ):
        """
        Initialize the GradientBoostingClassifierOptimized.

        Args:
            n_estimators (int): The number of boosting stages to run. Default is 100.
            learning_rate (float): The contribution of each tree. Default is 0.1.
            max_depth (int): The maximum depth of the trees. Default is 3.
            min_samples_split (int): The minimum number of samples required to split a node. Default is 2.
            loss (str or callable): The loss function to be used. Default is "logloss".
            verbose (bool): If True, prints progress during training. Default is False.
            boosting_type (str): The boosting type ("goss" for Gradient-based One-Side Sampling or "gbdt" for standard boosting). Default is "goss".
            alpha (float): The fraction of high gradient data used in GOSS. Default is 0.2.
            beta (float): The fraction of low gradient data sampled in GOSS. Default is 0.1.
            apply_efb (bool): If True, applies Exclusive Feature Bundling (EFB) to the data. Default is False.
            K (int): The conflict threshold for EFB bundling. Default is 1.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.boosting_type = boosting_type
        self.alpha = alpha
        self.beta = beta
        self.base_pred_ = None
        self.trees_ = []
        self.apply_efb = apply_efb
        self.K = K

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

    def _goss_sampling(self, X, residuals):
        """
        Perform Gradient-based One-Side Sampling (GOSS) to select a subset of samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            residuals (np.ndarray): Residuals of the current prediction.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Subsampled feature matrix and residuals.
        """
        gradients = np.abs(residuals)
        high_grad_indices = gradients >= np.percentile(gradients, 100 * (1 - self.alpha))
        low_grad_indices = ~high_grad_indices

        high_grad_X = X[high_grad_indices]
        high_grad_residuals = residuals[high_grad_indices]

        low_grad_indices = np.random.choice(
            np.where(low_grad_indices)[0],
            size=int(np.sum(low_grad_indices) * self.beta),
            replace=False
        )
        low_grad_X = X[low_grad_indices]
        low_grad_residuals = residuals[low_grad_indices]

        return np.vstack((high_grad_X, low_grad_X)), \
            np.concatenate((high_grad_residuals, low_grad_residuals))

    def _greedy_bundling(self, x, k):
        """
        Greedy bundling algorithm for Exclusive Feature Bundling (EFB).

        Args:
            x (np.ndarray): The feature matrix.
            k (int): The conflict threshold for bundling.

        Returns:
            list: A list of feature bundles.
        """
        n_rows = x.shape[0]
        n_cols = x.shape[1]
        conflict_count_matrix = np.zeros((n_cols, n_cols))

        for feature_i in range(n_cols):
            for feature_j in range(feature_i + 1, n_cols):
                conflict_count_matrix[feature_i, feature_j] = len(np.where(x[:, feature_i] * x[:, feature_j] > 0)[0])

        upper_triangle_indices = np.triu_indices(n_cols, 1)
        lower_triangle_indices = (upper_triangle_indices[1], upper_triangle_indices[0])
        conflict_count_matrix[lower_triangle_indices] = conflict_count_matrix[upper_triangle_indices]

        feature_degrees = conflict_count_matrix.sum(axis=0)
        sorted_feature_indices = np.argsort(feature_degrees)[::-1]

        bundles = []
        bundles_conflict = []
        for feature_index in sorted_feature_indices:
            need_new_bundle = True
            for bundle_index in range(len(bundles)):
                count_conflict = conflict_count_matrix[bundles[bundle_index][-1], feature_index]
                if count_conflict + bundles_conflict[bundle_index] <= k:
                    bundles[bundle_index].append(feature_index)
                    bundles_conflict[bundle_index] += count_conflict
                    need_new_bundle = False
                    break

            if need_new_bundle:
                bundles.append([feature_index])
                bundles_conflict.append(0.)
        return bundles

    def _merge_features(self, x, bundles):
        """
        Merge exclusive features into single features based on bundles.

        Args:
            x (np.ndarray): Feature matrix.
            bundles (list): A list of feature bundles.

        Returns:
            np.ndarray: The transformed feature matrix with merged features.
        """
        x_efb = np.zeros((x.shape[0], len(bundles)), dtype=float)

        for bundle_index, bundle in enumerate(bundles):
            x_bundle = x[:, bundle]
            max_values = x_bundle.max(axis=1)
            for _, idx in enumerate(bundle):
                x_efb[:, bundle_index] += x[:, idx] * (x[:, idx] == max_values)

        return x_efb

    def _apply_efb(self, X):
        """
        Apply Exclusive Feature Bundling (EFB) to the data.

        Args:
            X (np.ndarray): The feature matrix.

        Returns:
            np.ndarray: The transformed feature matrix after EFB.
        """
        bundles = self._greedy_bundling(X, self.K)
        return self._merge_features(X, bundles)

    def fit(self, X, y):
        """
        Fit the GradientBoostingClassifierOptimized model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target binary labels of shape (n_samples,).
        """
        if self.apply_efb:
            X = self._apply_efb(X)

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

            if self.boosting_type == "goss":
                X_sample, residuals_sample = self._goss_sampling(X, residuals)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, \
                min_samples_split=self.min_samples_split)
            if self.boosting_type == "goss":
                tree.fit(X_sample, residuals_sample)
            else:
                tree.fit(X, residuals)

            leaf_nodes = self._get_leaf_nodes(tree.tree_)
            for leaf in leaf_nodes:
                indices = self._get_samples_in_leaf(leaf, tree.tree_, X)
                gamma = self._calculate_gamma(y[indices], expit(y_pred_log_odds[indices]))
                y_pred_log_odds[indices] += self.learning_rate * gamma

            self.trees_.append(tree)

    def _get_leaf_nodes(self, node: Node) -> list:
        """
        Get all leaf nodes from a decision tree.

        Args:
            node (Node): The root node of the decision tree.

        Returns:
            list: A list of all leaf nodes.
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
        Get the indices of samples that belong to a specific leaf node.

        Args:
            leaf (Node): The leaf node.
            tree (Node): The decision tree from which the leaf belongs.
            X (np.ndarray): The feature matrix.

        Returns:
            np.ndarray: The indices of the samples in the leaf node.
        """
        indices = []
        for i, x in enumerate(X):
            node = tree  # Traverse this particular tree, not self.tree_
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
        Predict the binary class labels for new data.

        Args:
            X (np.ndarray): The feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted binary labels of shape (n_samples,).
        """
        predictions_log_odds = np.full(len(X), self.base_pred_)
        for tree in self.trees_:
            tree_predictions = tree.predict(X)
            if tree_predictions.shape[0] != len(X):
                raise ValueError("Mismatch between number of samples in X and tree predictions.")
            predictions_log_odds += self.learning_rate * tree_predictions
        proba = expit(predictions_log_odds)
        return (proba >= 0.5).astype(int)
