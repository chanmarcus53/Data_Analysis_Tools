"""
The modeling/trees.py file contains functions for building and evaluating decision tree models, 
including functions for training a decision tree classifier, Random Forest, and Gradient Boosting models,
as well as functions for performing cross-validation and calculating feature importance.
Last updated: 2026-04-29
By: Marcus Chan
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from performance_review.review import cross_validate_model, calculate_feature_importance, model_review


def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree Classifier on the given training data.
    Parameters:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training labels.

    Returns:
    - model: The trained Decision Tree Classifier model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier on the given training data.
    Parameters:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training labels.

    Returns:
    - model: The trained Random Forest Classifier model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """
    Trains a Gradient Boosting Classifier on the given training data.
    Parameters:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training labels.

    Returns:
    - model: The trained Gradient Boosting Classifier model.
    """
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


