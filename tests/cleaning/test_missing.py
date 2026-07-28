from toolbox.cleaning.missing import handle_missing, _flag, _drop, _impute_simple, _impute_knn, _impute_ffill, _impute_bfill
import pandas as pd
import pytest

class TestFlag:
    def test_flag_adds_column(self, null_df, audit_trail):
        # null_df and audit_trail come from conftest.py automatically
        result = _flag(null_df, "age", audit_trail)
        assert "age_missing" in result.columns

    def test_flag_correct_values(self, null_df, audit_trail):
        result = _flag(null_df, "age", audit_trail)
        # null_df has None in age at index 1
        # so age_missing should be True at index 1 and False elsewhere
        assert result["age_missing"].iloc[1] == True
        assert result["age_missing"].iloc[0] == False

    def test_flag_does_not_modify_original(self, null_df, audit_trail):
        # remember _flag works on a copy — original should be unchanged
        original_columns = list(null_df.columns)
        _flag(null_df, "age", audit_trail)
        assert list(null_df.columns) == original_columns

    def test_flag_logs_to_audit(self, null_df, audit_trail):
        _flag(null_df, "age", audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "flag"
        assert audit_trail.trail[0]["column"] == "age"

class TestDrop:
    def test_drop_rows_with_nulls(self, null_df, audit_trail):
        result = _drop(null_df, "age", audit=audit_trail)
        assert len(result) == len(null_df) - 1

    def test_drop_does_not_modify_original(self, null_df, audit_trail):
        original_len = len(null_df)
        _drop(null_df, "age", audit=audit_trail)
        assert len(null_df) == original_len

    def test_drop_with_threshold(self, null_df, audit_trail):
        # threshold=0.5 means drop rows with more than 50% nulls
        result = _drop(null_df, "age", threshold=0.5, audit=audit_trail)
        assert len(result) <= len(null_df)

    def test_drop_column_with_nulls(self, null_df, audit_trail):
        # axis=1 drops the column itself since age has nulls
        result = _drop(null_df, "age", axis=1, audit=audit_trail)
        assert "age" not in result.columns

    def test_drop_logs_to_audit(self, null_df, audit_trail):
        _drop(null_df, "age", audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["column"] == "age"

    def test_drop_no_nulls_unchanged(self, clean_df, audit_trail):
        # dropping from a clean DataFrame should return same length
        result = _drop(clean_df, "age", audit=audit_trail)
        assert len(result) == len(clean_df)

class TestImputeSimple:
    def test_impute_mean(self, null_df, audit_trail):
        result = _impute_simple(null_df, "age", "mean", audit=audit_trail)
        assert result["age"].isnull().sum() == 0

    def test_impute_median(self, null_df, audit_trail):
        result = _impute_simple(null_df, "age", "median", audit=audit_trail)
        assert result["age"].isnull().sum() == 0

    def test_impute_mode(self, null_df, audit_trail):
        result = _impute_simple(null_df, "age", "mode", audit=audit_trail)
        assert result["age"].isnull().sum() == 0

    def test_impute_constant(self, null_df, audit_trail):
        result = _impute_simple(null_df, "age", "constant", value=99, audit=audit_trail)
        assert result["age"].isnull().sum() == 0
        assert (result["age"] == 99).sum() == 1

    def test_impute_constant_no_value_raises(self, null_df, audit_trail):
        with pytest.raises(ValueError):
            _impute_simple(null_df, "age", "constant", audit=audit_trail)

    def test_impute_does_not_modify_original(self, null_df, audit_trail):
        original_null_count = null_df["age"].isnull().sum()
        _impute_simple(null_df, "age", "mean", audit=audit_trail)
        assert null_df["age"].isnull().sum() == original_null_count

    def test_impute_logs_to_audit(self, null_df, audit_trail):
        _impute_simple(null_df, "age", "mean", audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "impute_simple"

class TestImputeKnn:
    def test_impute_knn(self, audit_trail):
        # numeric only DataFrame for KNN
        df = pd.DataFrame({
            "age": [25.0, None, 35.0],
            "salary": [50000.0, 60000.0, 70000.0]
        })
        result = _impute_knn(df, "age", audit=audit_trail)
        assert result["age"].isnull().sum() == 0

    def test_impute_knn_does_not_modify_original(self, audit_trail):
        df = pd.DataFrame({
            "age": [25.0, None, 35.0],
            "salary": [50000.0, 60000.0, 70000.0]
        })
        original_null_count = df["age"].isnull().sum()
        _impute_knn(df, "age", audit=audit_trail)
        assert df["age"].isnull().sum() == original_null_count

class TestImputeFfill:
    def test_impute_ffill(self, null_df, audit_trail):
        result = _impute_ffill(null_df, "age", audit=audit_trail)
        assert result["age"].isnull().sum() == 0
        assert result["age"].iloc[1] == 25

    def test_ffill_does_not_modify_original(self, null_df, audit_trail):
        original_null_count = null_df["age"].isnull().sum()
        _impute_ffill(null_df, "age", audit=audit_trail)
        assert null_df["age"].isnull().sum() == original_null_count

    def test_ffill_logs_to_audit(self, null_df, audit_trail):
        _impute_ffill(null_df, "age", audit=audit_trail)
        assert len(audit_trail) == 1

class TestImpulseBfill:
    def test_impute_bfill(self, null_df, audit_trail):
        result = _impute_bfill(null_df, "age", audit=audit_trail)
        assert result["age"].isnull().sum() == 0
        assert result["age"].iloc[1] == 35

    def test_bfill_does_not_modify_original(self, null_df, audit_trail):
        original_null_count = null_df["age"].isnull().sum()
        _impute_bfill(null_df, "age", audit=audit_trail)
        assert null_df["age"].isnull().sum() == original_null_count

    def test_bfill_logs_to_audit(self, null_df, audit_trail):
        _impute_bfill(null_df, "age", audit=audit_trail)
        assert len(audit_trail) == 1
    

class TestHandleMissing:
    def test_handle_missing_flag(self, null_df, audit_trail):
        result = handle_missing(null_df, "age", "flag", audit=audit_trail)
        assert "age_missing" in result.columns
        assert result["age_missing"].iloc[1] == True

    def test_handle_missing_drop(self, null_df, audit_trail):
        result = handle_missing(null_df, "age", "drop", audit=audit_trail)
        assert len(result) == len(null_df) - 1
        assert "age" in result.columns

    def test_unsupported_strategy_raises(self, null_df, audit_trail):
        with pytest.raises(ValueError):
            handle_missing(null_df, "age", "unsupported", audit=audit_trail)

    def test_creates_audit_trail_if_none(self, null_df):
        # should not crash when no audit trail passed
        result = handle_missing(null_df, "age", "median")
        assert result["age"].isnull().sum() == 0