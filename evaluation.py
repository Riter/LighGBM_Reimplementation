import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import expit
from scipy.sparse import random as sparse_random
from time import time

def create_sparse_dataset(n_samples=1000, n_features=20, density=0.1, random_state=42):
    """
    Creates a sparse dataset with binary labels.

    Args:
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of features in the dataset.
        density (float): Density of the sparse matrix.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the sparse feature matrix and the binary labels.
    """
    sparse_X = sparse_random(n_samples, n_features, density=density, format='csr', random_state=random_state)
    sparse_y = np.random.randint(0, 2, size=n_samples)
    return sparse_X, sparse_y

def fit_and_time_model(model, X, y, model_name, fit_times):
    """
    Fits a model and records the time taken.

    Args:
        model (object): The model to fit.
        X (array-like): Training data.
        y (array-like): Target labels.
        model_name (str): Name of the model.
        fit_times (dict): Dictionary to store fitting times.

    Returns:
        object: The fitted model.
    """
    start_time = time()
    model.fit(X, y)
    end_time = time()
    fit_time = end_time - start_time
    fit_times[model_name] = fit_time
    print(f"{model_name} Training Time: {fit_time:.4f} seconds")
    return model

def predict_and_evaluate(model, X, y_true, model_name, eval_times, use_proba=False):
    """
    Makes predictions with the model and evaluates its performance.

    Args:
        model (object): The model to use for predictions.
        X (array-like): Test data.
        y_true (array-like): True labels.
        model_name (str): Name of the model.
        eval_times (dict): Dictionary to store evaluation times.
        use_proba (bool): Whether to use predicted probabilities for evaluation.

    Returns:
        None
    """
    start_time = time()
    y_pred = model.predict(X)
    if use_proba:
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else expit(
            model.base_pred_ + np.sum([tree.predict(X) for tree in model.trees_], axis=0)
        )
    else:
        y_pred_proba = y_pred
    end_time = time()
    eval_time = end_time - start_time
    eval_times[model_name] = eval_time
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"{model_name} - Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}, Evaluation Time: {eval_time:.4f} seconds")
