from toolbox.cleaning.outliers import handle_outliers, _detect, _flag, _cap, _remove
import pandas as pd
import pytest

class TestDetect:
    def test_detect_outliers_iqr(self, outlier_df):
        outliers = _detect(outlier_df, "age", method="iqr")
        assert len(outliers) == len(outlier_df)
        assert outliers.sum() == 1
        assert outliers.iloc[3] == True
        assert outliers.iloc[0] == False

    def test_detect_outliers_zscore(self, outlier_df):
        outliers = _detect(outlier_df, "age", method="zscore", threshold=1.5)
        assert len(outliers) == len(outlier_df)
        assert outliers.sum() == 1
        assert outliers.iloc[3] == True
        assert outliers.iloc[0] == False

    def test_detect_unsupported_method_raises(self, outlier_df):
        with pytest.raises(ValueError):
            _detect(outlier_df, "age", method="unsupported")

class TestFlag:
    def test_flag_adds_column(self, outlier_df, audit_trail):
        mask = _detect(outlier_df, "age", method="iqr")
        result = _flag(outlier_df, "age", mask, audit_trail)
        assert "age_outlier" in result.columns
        assert result["age_outlier"].sum() == 1
        assert result["age_outlier"].iloc[3] == True
        assert result["age_outlier"].iloc[0] == False

    def test_flag_does_not_modify_original(self, outlier_df, audit_trail):
        original = outlier_df.copy()
        mask = _detect(outlier_df, "age", method="iqr")
        _flag(outlier_df, "age", mask, audit_trail)
        pd.testing.assert_frame_equal(outlier_df, original)

    def test_flag_logs_to_audit(self, outlier_df, audit_trail):
        mask = _detect(outlier_df, "age", method="iqr")
        _flag(outlier_df, "age", mask, audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "flag"
        assert audit_trail.trail[0]["column"] == "age"

class TestCap:
    def test_cap_retains_normal_values(self, outlier_df, audit_trail):
        original = outlier_df.copy()
        mask = _detect(outlier_df, "age", method="iqr")
        result = _cap(outlier_df, "age", mask, method="iqr", audit=audit_trail)
        assert result["age"].iloc[0] == original["age"].iloc[0]
        assert result["age"].iloc[1] == original["age"].iloc[1]
        assert result["age"].iloc[2] == original["age"].iloc[2]

    def test_cap_replaces_outliers(self, outlier_df, audit_trail):
        original = outlier_df.copy()
        mask = _detect(outlier_df, "age", method="iqr")
        result = _cap(outlier_df, "age", mask, method="iqr", audit=audit_trail)
        assert result["age"].iloc[3] != original["age"].iloc[3]
        assert result["age"].iloc[3] <= 67.5  # upper bound

    def test_cap_does_not_modify_original(self, outlier_df, audit_trail):
        original = outlier_df.copy()
        mask = _detect(outlier_df, "age", method="iqr")
        _cap(outlier_df, "age", mask, method="iqr", audit=audit_trail)
        pd.testing.assert_frame_equal(outlier_df, original)

    def test_cap_logs_to_audit(self, outlier_df, audit_trail):
        mask = _detect(outlier_df, "age", method="iqr")
        _cap(outlier_df, "age", mask, method="iqr", audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "cap"

    def test_cap_zscore(self, outlier_df, audit_trail):
        mask = _detect(outlier_df, "age", method="zscore", threshold=2)
        result = _cap(outlier_df, "age", mask, method="zscore", threshold=2, audit=audit_trail)
        mean = outlier_df["age"].mean()
        std = outlier_df["age"].std()
        upper_bound = mean + 2 * std
        assert result["age"].max() <= upper_bound

class TestRemove:
    def test_remove_drops_outliers(self, outlier_df, audit_trail):
        mask = _detect(outlier_df, "age", method="iqr")
        result = _remove(outlier_df, "age", mask, audit=audit_trail)
        assert len(result) == len(outlier_df) - 1

    def test_remove_correct_row_removed(self, outlier_df, audit_trail):
        mask = _detect(outlier_df, "age", method="iqr")
        result = _remove(outlier_df, "age", mask, audit=audit_trail)
        assert 100 not in result["age"].values
        assert 25 in result["age"].values

    def test_remove_does_not_modify_original(self, outlier_df, audit_trail):
        original = outlier_df.copy()
        mask = _detect(outlier_df, "age", method="iqr")
        _remove(outlier_df, "age", mask, audit=audit_trail)
        pd.testing.assert_frame_equal(outlier_df, original)

    def test_remove_logs_to_audit(self, outlier_df, audit_trail):
        mask = _detect(outlier_df, "age", method="iqr")
        _remove(outlier_df, "age", mask, audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "remove"

class TestHandleOutliers:
    def test_handle_outliers_flag(self, outlier_df, audit_trail):
        result = handle_outliers(outlier_df, "age", method="iqr", action="flag", audit=audit_trail)
        assert "age_outlier" in result.columns
        assert result["age_outlier"].sum() == 1

    def test_handle_outliers_cap(self, outlier_df, audit_trail):
        result = handle_outliers(outlier_df, "age", method="iqr", action="cap", audit=audit_trail)
        assert result["age"].iloc[3] != outlier_df["age"].iloc[3]
        assert result["age"].iloc[3] <= 67.5

    def test_handle_outliers_remove(self, outlier_df, audit_trail):
        result = handle_outliers(outlier_df, "age", method="iqr", action="remove", audit=audit_trail)
        assert len(result) == len(outlier_df) - 1
        assert 100 not in result["age"].values

    def test_unsupported_method_raises(self, outlier_df, audit_trail):
        with pytest.raises(ValueError):
            handle_outliers(outlier_df, "age", method="unsupported", action="flag", audit=audit_trail)

    def test_unsupported_action_raises(self, outlier_df, audit_trail):
        with pytest.raises(ValueError):
            handle_outliers(outlier_df, "age", method="iqr", action="unsupported", audit=audit_trail)

    def test_creates_audit_trail_if_none(self, outlier_df):
        result = handle_outliers(outlier_df, "age", method="iqr", action="flag")
        assert isinstance(result, pd.DataFrame)

    def test_handle_outliers_zscore(self, outlier_df, audit_trail):
        result = handle_outliers(outlier_df, "age", method="zscore", action="flag", threshold=2, audit=audit_trail)
        assert "age_outlier" in result.columns