"""
This file contain functions for encoding categorical variables in the dataset 
such as one-hot encoding and label encoding.

Last updated: 2026-04-27
By: Marcus Chan
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, get_dummies

def label_encoding(data, column, related_values=False):
    """
    Encodes a categorical column using label encoding.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    column (str): The name of the column to encode.

    Returns:
    pd.Series: The column with label encoded values.
    """
    if related_values:
        # If related_values is True, we will encode the column based on the unique values in the column
        # then the label encoder encode the values properly, so we will use get_dummies to ensure consistency
        return pd.get_dummies(data[column], prefix=column)
    else:
        # If related_values is False, we will encode the column based on the unique values in the entire dataset
        le = LabelEncoder()
        return le.fit_transform(data[column])

def one_hot_encoding(data, column):
    """
    Encodes a categorical column using one-hot encoding.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    column (str): The name of the column to encode.

    Returns:
    pd.DataFrame: A DataFrame with one-hot encoded values for the specified column.
    """
    ohe = OneHotEncoder(sparse=False)
    encoded_data = ohe.fit_transform(data[[column]])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out([column]))
    return encoded_df