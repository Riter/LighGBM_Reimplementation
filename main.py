import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier as SKLearnGBC
import xgboost as xgb
import lightgbm as lgb
from scipy.special import expit
from scipy.sparse import random as sparse_random
from time import time
from gradient_boosting_optimized import GradientBoostingClassifierOptimized
from gradient_boosting import GradientBoostingClassifier
from evaluation import create_sparse_dataset, fit_and_time_model, predict_and_evaluate

def main():
    """
    Main function to create a sparse dataset, fit models, and evaluate them.
    """
    # Generate a sparse dataset
    X_train, y_train = create_sparse_dataset(n_samples=700, n_features=20, density=0.05, random_state=42)
    X_test, y_test = create_sparse_dataset(n_samples=300, n_features=20, density=0.05, random_state=42)

    # Initialize classifiers
    simple_boosting_goss_and_efb = GradientBoostingClassifierOptimized(
        n_estimators=100, learning_rate=0.1, max_depth=3, verbose=True, boosting_type="goss", apply_efb=True
    )
    simple_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    sklearn_model = SKLearnGBC(n_estimators=100, learning_rate=0.1, max_depth=3)
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, verbose=0)

    # Fit models and record fitting times
    fit_times = {}
    simple_boosting_goss_and_efb = fit_and_time_model(simple_boosting_goss_and_efb, X_train.toarray(), y_train, "Optimized Boosting", fit_times)
    simple_boosting = fit_and_time_model(simple_boosting, X_train.toarray(), y_train, "Simple Boosting", fit_times)
    sklearn_model = fit_and_time_model(sklearn_model, X_train.toarray(), y_train, "sklearn GradientBoosting", fit_times)
    xgb_model = fit_and_time_model(xgb_model, X_train, y_train, "XGBoost", fit_times)
    lgb_model = fit_and_time_model(lgb_model, X_train, y_train, "LightGBM", fit_times)

    # Make predictions and evaluate models
    eval_times = {}
    predict_and_evaluate(simple_boosting_goss_and_efb, X_test.toarray(), y_test, "Optimized Boosting", eval_times, use_proba=True)
    predict_and_evaluate(simple_boosting, X_test.toarray(), y_test, "Simple Boosting", eval_times, use_proba=True)
    predict_and_evaluate(sklearn_model, X_test.toarray(), y_test, "sklearn GradientBoosting", eval_times, use_proba=True)
    predict_and_evaluate(xgb_model, X_test, y_test, "XGBoost", eval_times, use_proba=True)
    predict_and_evaluate(lgb_model, X_test, y_test, "LightGBM", eval_times, use_proba=True)

    # Print total computation times
    print("\nSummary of Computation Times:")
    for model_name in fit_times:
        print(f"{model_name} - Training Time: {fit_times[model_name]:.4f} seconds, Evaluation Time: {eval_times[model_name]:.4f} seconds")

if __name__ == "__main__":
    main()
