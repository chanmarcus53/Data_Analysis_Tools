"""
This module contains functions for imputing missing values in the dataset, 
such as mean imputation, median imputation, and standardKNN imputation.
Last updated: 2026-04-25
By: Marcus Chan

Note: All data imputation functions in this module are designed to handle numerical data in pandas DataFrames.
"""
import pandas as pd
from sklearn.impute import KNNImputer

def mean_imputation(data, column):
    """
    Imputes missing values in a column using the mean of the non-missing values.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    column (str): The name of the column to impute.

    Returns:
    pd.Series: The column with imputed values.
    """
    mean_value = data[column].mean()
    return data[column].fillna(mean_value)

def median_imputation(data, column):
    """
    Imputes missing values in a column using the median of the non-missing values.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    column (str): The name of the column to impute.

    Returns:
    pd.Series: The column with imputed values.
    """
    median_value = data[column].median()
    return data[column].fillna(median_value)

def knn_imputation(data, column, k=5):
    """
    Imputes missing values in a column using K-Nearest Neighbors (KNN) imputation.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    column (str): The name of the column to impute.
    k (int): The number of nearest neighbors to use for imputation.

    Returns:
    pd.Series: The column with imputed values.
    """
    imputer = KNNImputer(n_neighbors=k)
    imputed_data = imputer.fit_transform(data[[column]])
    return pd.Series(imputed_data.flatten(), index=data.index)