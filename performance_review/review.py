"""
This module contains the code for the performance review of the data analysis tools. 
It will be used to evaluate the performance of the tools and provide feedback for improvement.
Last updated: 2026-04-27
By: Marcus Chan
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

def model_review(model, X_data, y_data, splits=5):
    """
    Evaluates the performance of a given model on the test data and provides feedback for improvement.
    
    Parameters:
    - model: The machine learning model to be evaluated.
    - X_test: The test features.
    - y_test: The true labels for the test data.
    - splits: The number of cross-validation folds (default is 5).

    Returns:
    - dict: A dictionary containing the evaluation metrics and feedback for improvement.
    """

    kf = KFold(n_splits=splits, shuffle=True, random_state=55)

    store_metrics = []
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        rmse = root_mean_squared_error(y_test, predictions, squared=False)
        store_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'rmse': rmse
        })

    # Calculate average metrics across all folds
    avg_accuracy = sum(m['accuracy'] for m in store_metrics) / splits
    avg_precision = sum(m['precision'] for m in store_metrics) / splits
    avg_recall = sum(m['recall'] for m in store_metrics) / splits
    avg_f1 = sum(m['f1'] for m in store_metrics) / splits
    avg_rmse = sum(m['rmse'] for m in store_metrics) / splits

    feedback = {}
    if avg_accuracy < 0.7:
        feedback['accuracy'] = "The model's accuracy is below 70%. Consider improving the feature engineering or trying a different algorithm."
    else:
        feedback['accuracy'] = "The model's accuracy is good."

    if avg_precision < 0.7:
        feedback['precision'] = "The model's precision is below 70%. Consider addressing class imbalance or tuning hyperparameters."
    else:
        feedback['precision'] = "The model's precision is good."

    if avg_recall < 0.7:
        feedback['recall'] = "The model's recall is below 70%. Consider addressing class imbalance or tuning hyperparameters."
    else:
        feedback['recall'] = "The model's recall is good."

    if avg_f1 < 0.7:
        feedback['f1'] = "The model's F1 score is below 70%. Consider improving both precision and recall for better performance."
    else:
        feedback['f1'] = "The model's F1 score is good."

    if avg_rmse > 1.0:
        feedback['rmse'] = "The model's RMSE is above 1.0. Consider improving the model's predictions or trying a different algorithm."
    else:
        feedback['rmse'] = "The model's RMSE is good."

    return {
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'rmse': avg_rmse,
        'feedback': feedback
    }

def model_review_narrow(model, X, y):
    """
    Evaluates the performance of a given model on the test data and provides feedback for improvement.
    This is a narrower version of the model_review function that focuses on a single fold of the data for evaluation.
    Parameters:
    - model: The machine learning model to be evaluated.
    - X: The features of the dataset.
    - y: The true labels for the dataset.

    Returns:
    - dict: A dictionary containing the evaluation metrics and feedback for improvement.
    """
    model.fit(X, y)
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='weighted')
    recall = recall_score(y, predictions, average='weighted')
    f1 = f1_score(y, predictions, average='weighted')
    rmse = root_mean_squared_error(y, predictions, squared=False)
    feedback = {}
    if accuracy < 0.7:
        feedback['accuracy'] = "The model's accuracy is below 70%. Consider improving the feature engineering or trying a different algorithm."
    else:
        feedback['accuracy'] = "The model's accuracy is good."
    
    if precision < 0.7:
        feedback['precision'] = "The model's precision is below 70%. Consider addressing class imbalance or tuning hyperparameters."
    else:
        feedback['precision'] = "The model's precision is good."
    
    if recall < 0.7:
        feedback['recall'] = "The model's recall is below 70%. Consider addressing class imbalance or tuning hyperparameters."
    else:
        feedback['recall'] = "The model's recall is good."

    if f1 < 0.7:
        feedback['f1'] = "The model's F1 score is below 70%. Consider improving both precision and recall for better performance."
    else:
        feedback['f1'] = "The model's F1 score is good."

    if rmse > 1.0:
        feedback['rmse'] = "The model's RMSE is above 1.0. Consider improving the model's predictions or trying a different algorithm."
    else:
        feedback['rmse'] = "The model's RMSE is good."

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rmse': rmse,
        'feedback': feedback
    }
