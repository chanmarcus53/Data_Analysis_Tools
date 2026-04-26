"""
This module contains functions for exploring the dataset, such as visualizations and summary statistics.

Last updated: 2026-04-25
By: Marcus Chan
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(data, target_col):
    """
    Plots the distribution of the target variable.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    target_col (str): The name of the target column.

    Responses:
    A bar plot showing the distribution of the target variable.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=data)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.show()

def plot_correlation_heatmap(data):
    """
    Plots a correlation heatmap of the features in the dataset.
    
    Parameters:
    data (pd.DataFrame): The input dataset.

    Responses:
    A heatmap showing the correlation between features.
    """
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


def show_feature_types_and_missing_values(data):
    """
    Shows the feature types and the number of missing values for each feature in the dataset.
    
    Parameters:
    data (pd.DataFrame): The input dataset.

    Responses:
    A summary table showing the feature types and the count of missing values for each feature.
    """
    feature_types = data.dtypes
    missing_values = data.isnull().sum()
    summary = pd.DataFrame({'Feature Type': feature_types, 'Missing Values': missing_values})
    print(summary)

def plot_feature_distributions(data):
    """
    Plots the distributions of the features in the dataset.
    
    Parameters:
    data (pd.DataFrame): The input dataset.

    Responses:
    A series of plots showing the distribution of each feature in the dataset.
    """
    for column in data.columns:
        plt.figure(figsize=(8, 6))
        if data[column].dtype == 'object':
            sns.countplot(x=column, data=data)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
        else:
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        plt.show()