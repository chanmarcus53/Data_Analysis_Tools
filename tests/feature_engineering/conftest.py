import pytest
import pandas as pd

@pytest.fixture
def encoding_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "size": ["small", "medium", "large", "small", "large", "medium"],
        "color": ["red", "blue", "red", "green", "blue", "red"],
        "price": [10, 20, 30, 12, 28, 18]  # numeric target for target encoding
    })

@pytest.fixture
def binning_df():
    return pd.DataFrame({
        "id": [1,2,3,4,5,6,7,8,9,10],
        "age": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        "salary": [20000, 35000, 45000, 55000, 65000, 75000, 85000, 95000, 110000, 150000]
    })

@pytest.fixture
def datetime_df():
    return pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "value": [10, 20, 15, 25, 30, 20, 35, 40, 30, 45]
    })

@pytest.fixture
def importance_df():
    return pd.DataFrame({
        "feature_a": [1,2,3,4,5,6,7,8,9,10],
        "feature_b": [10,9,8,7,6,5,4,3,2,1],
        "feature_c": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "target": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    })