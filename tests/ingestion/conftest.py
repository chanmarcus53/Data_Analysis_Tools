import pytest
import pandas as pd

@pytest.fixture
def clean_df():
    """A simple well-formed DataFrame for happy path tests"""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "status": ["active", "inactive", "active"]
    })

@pytest.fixture
def null_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", None],
        "age": [25, None, 35],
        "status": ["active", "inactive", "active"]
    })

@pytest.fixture
def malformed_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "ag": [25, 30, 35],
        "stat": ["active", "inactive", "active"]
    })

@pytest.fixture
def sample_api_response():
    """Simulates a typical API JSON response"""
    return {
        "results": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]
    }

@pytest.fixture
def sample_schema():
    """A schema to validate against clean_df"""
    return {
        "columns": ["id", "name", "age", "status"],
        "dtypes": {"id": "int64", "age": "int64"},
        "non_nullable": ["id"],
        "value_sets": {"status": ["active", "inactive"]}
    }

@pytest.fixture
def high_null_df():
    """DataFrame with high null percentage for testing issue detection"""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": [None, None, None],
        "age": [None, None, None],
        "status": [None, None, None]
    })

@pytest.fixture
def single_value_df():
    """DataFrame where one column has a single unique value"""
    return pd.DataFrame({
        "id": [1,2,3],
        "name": ["Alice", "Alice", "Alice"],
        "age": [25, 25, 25],
        "status": ["active", "active", "active"]
    })

@pytest.fixture
def duplicate_df():
    """DataFrame with duplicate rows for testing issue detection"""
    return pd.DataFrame({
        "id": [1, 1, 2],
        "name": ["Alice", "Alice", "Bob"],
        "age": [25, 25, 30],
        "status": ["active", "active", "inactive"]
    })
