"""
This module contains code for tuning machine learning models, such as determining the optimal number of neighbors for KNN classification.
Last updated: 2026-04-30
By: Marcus Chan
"""
from sklearn.model_selection import GridSearchCV, KFold


# Grid search
def grid_search(model, param_grid, X_train, y_train):
    """
    Performs grid search to find the best hyperparameters for a given model.
    
    Parameters:
    - model: The machine learning model for which to perform grid search.
    - param_grid: A dictionary specifying the hyperparameters and their corresponding values to be evaluated.
    - X_train: The training features.
    - y_train: The training labels.

    Returns:
    - best_params: The best hyperparameters found by grid search.
    - best_score: The best score achieved with the best hyperparameters.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_params, best_score

# Bayesian optimization
def bayesian_optimization(model, param_space, X_train, y_train):
    """
    Performs Bayesian optimization to find the best hyperparameters for a given model.
    
    Parameters:
    - model: The machine learning model for which to perform Bayesian optimization.
    - param_space: A dictionary specifying the hyperparameters and their corresponding ranges to be evaluated.
    - X_train: The training features.
    - y_train: The training labels.

    Returns:
    - best_params: The best hyperparameters found by Bayesian optimization.
    - best_score: The best score achieved with the best hyperparameters.
    """
    from skopt import BayesSearchCV
    
    bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, cv=5, scoring='accuracy')
    bayes_search.fit(X_train, y_train)
    
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    
    return best_params, best_score