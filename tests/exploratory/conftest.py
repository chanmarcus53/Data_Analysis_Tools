import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve",
                 "Frank", "Grace", "Hank", "Iris", "Jack"],
        "grade": [50, 35, 40, 40, 60, 70, 90, 85, None, 55],
        "salary": [30000, 45000, 52000, 48000, 61000,
                   75000, 90000, 85000, 72000, 95000],
        "school": ["A", "A", "B", "B", "C", "C", "C", "A", "B", "C"]
    })

@pytest.fixture
def categorical_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
        "grade": [50, 35, 40, 40, 60, 70, 90],
        "school": ["A", "A", "B", "B", "C", "C", "C"]
    })

@pytest.fixture
def time_series_df():
    return pd.DataFrame({
        "id": range(1, 13),
        "grade": [50, 55, 48, 60, 62, 58, 70, 72, 68, 80, 85, 78],
        "quarter": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2",
                    "Q3", "Q3", "Q3", "Q4", "Q4", "Q4"]
    })

@pytest.fixture
def eda_result(sample_df):
    from toolbox.exploratory.eda import _collect_results
    return _collect_results(sample_df)

