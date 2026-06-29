import pandas as pd
import pytest

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
def sample_schema():
    """A schema to validate against clean_df"""
    return {
        "columns": ["id", "name", "age", "status"],
        "dtypes": {"id": "int64", "age": "int64"},
        "non_nullable": ["id"],
        "value_sets": {"status": ["active", "inactive"]}
    }