"""
This modeling/KNN_processing.py file contains functions for processing data using K-Nearest Neighbors (KNN) algorithm,
such as KNN imputation for missing values and KNN classification for predictive modeling.
Last updated: 2026-04-27
By: Marcus Chan
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

def knn_classification(X_train, y_train, X_test, k=5):
    """
    Performs K-Nearest Neighbors (KNN) classification on the given training and test data.
    
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training labels.
    X_test (pd.DataFrame): The test features to predict.
    k (int): The number of nearest neighbors to use for classification.

    Returns:
    np.ndarray: The predicted labels for the test data.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

def knn_regression(X_train, y_train, X_test, k=5):
    """
    Performs K-Nearest Neighbors (KNN) regression on the given training and test data.
    
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target values.
    X_test (pd.DataFrame): The test features to predict.
    k (int): The number of nearest neighbors to use for regression.

    Returns:
    np.ndarray: The predicted target values for the test data.
    """
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

def determine_optimal_k(X_train, y_train, X_test, y_test, max_k=20):
    """
    Determines the optimal number of neighbors (k) for KNN classification by evaluating the accuracy on the test set.
    
    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training labels.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The true labels for the test data.
    max_k (int): The maximum number of neighbors to evaluate.

    Returns:
    int: The optimal number of neighbors (k) that yields the highest accuracy on the test set.
    """
    from sklearn.metrics import accuracy_score

    best_k = 1
    best_accuracy = 0

    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k
