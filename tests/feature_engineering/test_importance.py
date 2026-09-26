from toolbox.feature_engineering.importance import rank_features, _detect_task, _random_forest_importance, _correlation_importance, _mutual_information
import pytest
import pandas as pd

class TestDetectTask:
    def test_detect_task_fa(self, importance_df):
        result = _detect_task(importance_df["feature_a"])
        assert result == "regression"

    def test_detect_task_fb(self, importance_df):
        result = _detect_task(importance_df["feature_b"])
        assert result == "classification"

    def test_detect_task_fd(self, importance_df):
        result = _detect_task(importance_df["feature_d"])
        assert result == "classification"

class TestRandomForestImportance:
    def test_dataframe_unchanged(self, importance_df):
        original = importance_df.copy()
        result = _random_forest_importance(importance_df, "target", task="regression")
        pd.testing.assert_frame_equal(original, importance_df)

    def test_return_correct_columns(self, importance_df):
        result = _random_forest_importance(importance_df, "target", task="regression")
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_sorted_descending(self, importance_df):
        result = _random_forest_importance(importance_df, "target", task="regression")
        assert result["importance"].iloc[0] >= result["importance"].iloc[-1]

    def test_top_feature_is_feature_a(self, importance_df):
        result = _random_forest_importance(importance_df, "target", task="regression")
        assert result["feature"].iloc[0] == "feature_a"

    def test_no_numeric_features_raises(self):
        df = pd.DataFrame({
            "category": ["a", "b", "c", "d", "e"],
            "target": [1, 2, 3, 4, 5]
        })
        with pytest.raises(ValueError):
            _random_forest_importance(df, "target", task="regression")

    def test_classification_task(self, importance_df):
        result = _random_forest_importance(importance_df, "feature_c", task="classification")
        assert "feature" in result.columns
        assert "importance" in result.columns
        assert result["importance"].iloc[0] >= result["importance"].iloc[-1]

    def test_randomforest_Xempty(self, importance_df):
        df = pd.DataFrame({
            "category": ["a", "b", "c", "d", "e",
                        "f", "g", "h", "i", "j"],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        with pytest.raises(ValueError):
            _random_forest_importance(df, "target", task="regression")


class TestCorrelationImportance:
    def test_dataframe_unchanged(self, importance_df):
        original = importance_df.copy()
        result = _correlation_importance(importance_df, "target")
        pd.testing.assert_frame_equal(original, importance_df)

    def test_returns_correct_columns(self, importance_df):
        result = _correlation_importance(importance_df, "target")
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_sorted_descending(self, importance_df):
        result = _correlation_importance(importance_df, "target")
        assert result["importance"].iloc[0] >= result["importance"].iloc[-1]

    def test_top_feature_is_feature_a(self, importance_df):
        result = _correlation_importance(importance_df, "target")
        assert result["feature"].iloc[0] == "feature_a"

    def test_uses_absolute_correlation(self, importance_df):
        result = _correlation_importance(importance_df, "target")
        assert (result["importance"] >= 0).all()

    def test_constant_column_gets_zero_importance(self):
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "constant": [1, 1, 1, 1, 1],  # constant column
            "target": [2, 4, 6, 8, 10]
        })
        result = _correlation_importance(df, "target")
        constant_row = result[result["feature"] == "constant"]
        assert constant_row["importance"].iloc[0] == 0

    def test_no_numeric_features_raises(self):
        df = pd.DataFrame({
            "category": ["a", "b", "c", "d", "e"],
            "target": [1, 2, 3, 4, 5]
        })
        with pytest.raises(ValueError):
            _correlation_importance(df, "target")

    def test_features_b_is_negative_correlation(self, importance_df):
        result = _correlation_importance(importance_df, "target")
        feature_b_row = result[result["feature"] == "feature_b"]
        assert feature_b_row["importance"].iloc[0] >= 0


class TestMutualInformation:
    def test_dataframe_unchanced(self, importance_df):
        original = importance_df.copy()
        result = _mutual_information(importance_df, "target", "regression")
        pd.testing.assert_frame_equal(original, importance_df)

    def test_returns_correct_columns(self, importance_df):
        result = _mutual_information(importance_df, "target", "regression")
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_sorted_descending(self, importance_df):
        result = _mutual_information(importance_df, "target", "regression")
        assert result["importance"].iloc[0] >= result["importance"].iloc[-1]

    def test_classification_task(self, importance_df):
        result = _mutual_information(importance_df, "feature_c", "classification")
        assert "feature" in result.columns
        assert "importance" in result.columns
        assert result["importance"].iloc[0] >= result["importance"].iloc[-1]

    def test_regression_task(self, importance_df):
        result = _mutual_information(importance_df, "target", "regression")
        assert result["importance"].iloc[0] >= result["importance"].iloc[-1]
        assert result["feature"].iloc[0] in ["feature_a", "feature_b"]

    def test_no_numeric_features_raises(self):
        df = pd.DataFrame({
            "category": ["a", "b", "c", "d", "e"],
            "target": [1, 2, 3, 4, 5]
        })
        with pytest.raises(ValueError):
            _mutual_information(df, "target", "regression")

    def test_importance_values_non_negative(self, importance_df):
        # mutual information is always >= 0
        result = _mutual_information(importance_df, "target", "regression")
        assert (result["importance"] >= 0).all()
        

class TestRankFeatures:
    def test_missing_target_raises(self, importance_df):
        with pytest.raises(ValueError):
            rank_features(importance_df, "nonexistent_column")

    def test_auto_method(self, importance_df):
        result = rank_features(importance_df, "target")
        assert "feature" in result.columns
        assert "importance" in result.columns
        assert result["feature"].iloc[0] == "feature_a"

    def test_random_forest_method(self, importance_df):
        result = rank_features(importance_df, "target", method="random_forest")
        assert "feature" in result.columns
        assert result["feature"].iloc[0] == "feature_a"

    def test_correlation_method(self, importance_df):
        result = rank_features(importance_df, "target", method="correlation")
        assert "feature" in result.columns
        assert result["feature"].iloc[0] == "feature_a"

    def test_mutual_information_method(self, importance_df):
        result = rank_features(importance_df, "target", method="mutual_information")
        assert "feature" in result.columns
        assert result["feature"].iloc[0] == "feature_a"

    def test_unsupported_method_raises(self, importance_df):
        with pytest.raises(ValueError):
            rank_features(importance_df, "target", method="unsupported")

    def test_result_sorted_descending(self, importance_df):
        result = rank_features(importance_df, "target")
        assert result["importance"].iloc[0] >= result["importance"].iloc[-1]

    def test_result_is_dataframe(self, importance_df):
        result = rank_features(importance_df, "target")
        assert isinstance(result, pd.DataFrame)

    def test_auto_detects_regression(self, importance_df):
        # target has many unique values so should pick regression method
        result = rank_features(importance_df, "target", method="auto")
        assert "feature" in result.columns

    def test_auto_detects_classification(self, importance_df):
        # feature_c has few unique values so should pick classification method
        result = rank_features(importance_df, "feature_c", method="auto")
        assert "feature" in result.columns

    def test_logs_to_audit(self, importance_df, audit_trail):
        rank_features(importance_df, "target", audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "rank_features"
        assert audit_trail.trail[0]["column"] == "target"

    def test_excludes_string_columns(self, importance_df):
        # feature_d is a string column and should be excluded from ranking
        result = rank_features(importance_df, "target")
        assert "feature_d" not in result["feature"].values