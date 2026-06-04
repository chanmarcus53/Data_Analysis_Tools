from toolbox.ingestion.validator import validate, _check_columns, _check_dtypes, _check_nulls, _check_value_sets
import pytest

def make_results():
    """Helper to create a fresh results dict for each test"""
    return {"passed": [], "warnings": [], "failed": []}

class TestCheckColumns:
    def test_valid_columns(self, clean_df):
        results = make_results()
        _check_columns(clean_df, ["id", "name", "age", "status"], results)
        assert len(results["failed"]) == 0

    def test_missing_column(self, malformed_df):
        results = make_results()
        _check_columns(malformed_df, ["id", "name", "age", "status"], results)
        assert len(results["failed"]) == 1
        assert "age" in results["failed"][0] or "status" in results["failed"][0]

    def test_extra_columns_are_warnings(self, clean_df):
        results = make_results()
        _check_columns(clean_df, ["id", "name"], results)
        assert len(results["warnings"]) == 1

class TestCheckDtypes:
    def test_valid_dtypes(self, clean_df):
        results = make_results()
        _check_dtypes(clean_df, {"id": "int64", "age": "int64"}, results)
        assert len(results["failed"]) == 0

    def test_invalid_dtype(self, clean_df):
        results = make_results()
        _check_dtypes(clean_df, {"id": "object"}, results)
        assert len(results["failed"]) == 1
        assert "id" in results["failed"][0]

    def test_missing_column_dtype(self, clean_df):
        results = make_results()
        _check_dtypes(clean_df, {"nonexistent": "int64"}, results)
        assert len(results["failed"]) == 1

class TestCheckNulls:
    def test_no_nulls(self, clean_df):
        results = make_results()
        _check_nulls(clean_df, ["id", "name"], results)
        assert len(results["failed"]) == 0

    def test_null_in_non_nullable(self, null_df):
        results = make_results()
        _check_nulls(null_df, ["age"], results)
        assert len(results["failed"]) == 1
        assert "age" in results["failed"][0]

    def test_missing_column_null_check(self, clean_df):
        results = make_results()
        _check_nulls(clean_df, ["nonexistent"], results)
        assert len(results["failed"]) == 1

class TestCheckValueSets:
    def test_valid_values(self, clean_df):
        results = make_results()
        _check_value_sets(clean_df, {"status": ["active", "inactive"]}, results)
        assert len(results["failed"]) == 0

    def test_invalid_values(self, clean_df):
        results = make_results()
        _check_value_sets(clean_df, {"status": ["active"]}, results)
        assert len(results["failed"]) == 1
        assert "inactive" in results["failed"][0]

class TestValidate:
    def test_full_pipeline_passes(self, clean_df, sample_schema):
        results = validate(clean_df, sample_schema)
        assert len(results["failed"]) == 0

    def test_no_schema_returns_empty(self, clean_df):
        results = validate(clean_df)
        assert results == {"passed": [], "warnings": [], "failed": []}

    def test_full_pipeline_catches_failures(self, malformed_df, sample_schema):
        results = validate(malformed_df, sample_schema)
        assert len(results["failed"]) > 0