import pytest
import pandas as pd

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

@pytest.fixture
def profile_result():
    return {
        "shape": (3, 4),
        "memory": 0.002,
        "columns": {
            "id": {
                "dtype": "int64",
                "count": 3,
                "null_count": 0,
                "null_pct": 0.0,
                "mean": 2.0,
                "std": 1.0,
                "min": 1.0,
                "max": 3.0,
                "skewness": 0.0
            },
            "status": {
                "dtype": "object",
                "count": 3,
                "null_count": 0,
                "null_pct": 0.0,
                "unique_count": 2,
                "top_values": {"active": 2, "inactive": 1}
            }
        },
        "warnings": ["Column 'id' has high null percentage: 60.0%"]
    }