from toolbox.ingestion.profiler import profile, _get_shape, _get_memory, _profile_columns, _profile_numeric, _profile_categorical, _detect_issues
import pytest
import pandas as pd

class TestGetShape:
    def test_get_shape(self, clean_df):
        assert _get_shape(clean_df) == [3, 4]

class TestGetMemory:
    def test_get_memory(self,  clean_df):
        result = _get_memory(clean_df)
        assert isinstance(result, float)
        assert result > 0  # don't assert exact bytes, it varies by system
        
class TestProfileNumeric:
    def test_profile_numeric(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = _profile_numeric(series)
        assert result["count"] == 5
        assert result["null_count"] == 0
        assert result["mean"] == pytest.approx(3.0)
        assert result["std"] == pytest.approx(1.5811, rel=1e-3)
        assert result["skewness"] == pytest.approx(0.0, abs=1e-4)
        assert result["dtype"] == "int64"

class TestProfileCategorical:
    def test_profile_categorical(self):
        import pandas as pd
        series = pd.Series(['a', 'b', 'c', 'a', 'd'])
        result = _profile_categorical(series)
        assert result["count"] == 5
        assert result["unique_count"] == 4
        assert result["null_count"] == 0
        assert result["top_values"]["a"] == 2

class TestProfileColumns:
    def test_profile_columns(self, clean_df):
        result = _profile_columns(clean_df)
        assert "id" in result
        assert "status" in result
        assert "mean" in result["id"]          # numeric
        assert "unique_count" in result["status"]  # categorical

    def test_all_columns_profiled(self, clean_df):
        result = _profile_columns(clean_df)
        assert len(result) == len(clean_df.columns)

class TestDetectIssues:
    def test_detect_issues(self, high_null_df):
        column_profile = _profile_columns(high_null_df)
        issues = _detect_issues(high_null_df, column_profile)
        assert any("high percentage of null values" in issue for issue in issues)

    def test_detect_issues_duplicates(self, duplicate_df):
        column_profile = _profile_columns(duplicate_df)
        issues = _detect_issues(duplicate_df, column_profile)
        assert any("duplicate rows detected" in issue for issue in issues)

    def test_detect_issues_single_value(self, single_value_df):
        column_profile = _profile_columns(single_value_df)
        issues = _detect_issues(single_value_df, column_profile)
        assert any("only one unique value" in issue for issue in issues)

class TestProfile:
    def test_profile_returns_dict(self, clean_df):
        result = profile(clean_df)
        assert isinstance(result, dict)

    def test_profile_has_all_keys(self, clean_df):
        result = profile(clean_df)
        assert "shape" in result
        assert "memory" in result
        assert "columns" in result
        assert "warnings" in result

    def test_profile_shape_matches_df(self, clean_df):
        result = profile(clean_df)
        assert result["shape"] == [3, 4]

    def test_profile_warnings_is_list(self, clean_df):
        result = profile(clean_df)
        assert isinstance(result["warnings"], list)

    def test_profile_columns_match_df(self, clean_df, sample_schema):
        result = profile(clean_df)
        assert set(sample_schema["columns"]).issubset(set(result["columns"].keys()))