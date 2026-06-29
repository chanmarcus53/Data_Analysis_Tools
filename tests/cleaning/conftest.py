import pandas as pd
import pytest
from  toolbox.cleaning.audit import AuditTrail

@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 100, 45],
        "status": ["active", "inactive", "active", "inactive", "active"]
    })

@pytest.fixture
def audit_trail():
    return AuditTrail()

@pytest.fixture
def sample_pipeline_steps():
    return [
        {"step": "handle_missing", "column": "age", "strategy": "median"},
        {"step": "handle_outliers", "column": "age", "method": "iqr", "action": "flag"}
    ]