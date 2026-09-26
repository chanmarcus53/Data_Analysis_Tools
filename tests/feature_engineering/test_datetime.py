from toolbox.feature_engineering.datetime import extract_datetime_features, _extract_basic, _extract_lag, _extract_seasonality
import pytest
import pandas as pd
import numpy as np

class TestExtractBasic:
    def test_dataframe_unchanged(self, datetime_df):
        original = datetime_df.copy()
        result = _extract_basic(datetime_df, "date", "month")
        pd.testing.assert_frame_equal(original, datetime_df)

    def test_extractbasic_year(self, datetime_df):
        result = _extract_basic(datetime_df, "date", "year")
        assert "date_year" in result.columns
        assert result["date_year"][0] == 2024

    def test_extractbasic_month(self, datetime_df):
        result = _extract_basic(datetime_df, "date", "month")
        assert "date_month" in result.columns
        assert result["date_month"][0] == 1

    def test_extractbasic_day(self, datetime_df):
        result = _extract_basic(datetime_df, "date", "day")
        assert "date_day" in result.columns
        assert result["date_day"][0] == 1

    def test_extractbasic_unknown_feature(self, datetime_df):
        with pytest.raises(ValueError):
            result = _extract_basic(datetime_df, "date", "hello")


class TestExtractLag:
    def test_dataframe_unchanged(self, datetime_df):
        original = datetime_df.copy()
        result = _extract_lag(datetime_df, "date", [7])
        pd.testing.assert_frame_equal(original, datetime_df)

    def test_extractlag_basic(self, datetime_df):
        result = _extract_lag(datetime_df, "date", [2])
        assert "date_lag_2" in result.columns

    def test_extractlag_simple(self, datetime_df):
        result = _extract_lag(datetime_df, "value", [1])
        assert "value_lag_1" in result.columns
        assert pd.isna(result["value_lag_1"].iloc[0])
        assert result["value_lag_1"].iloc[1] == 10

    def test_extractlag_multiple(self, datetime_df):
        result = _extract_lag(datetime_df, "value", [1,2,3])
        assert "value_lag_1" in result.columns
        assert "value_lag_2" in result.columns
        assert "value_lag_3" in result.columns

    def test_extractlag_logs_to_audit(self, datetime_df, audit_trail):
        result = _extract_lag(datetime_df, "date", [7], audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "lag"
        assert audit_trail.trail[0]["column"] == "date"


class TestExtractSeasonality:
    def test_dataframe_unchanged(self, datetime_df):
        original = datetime_df.copy()
        result = _extract_seasonality(datetime_df, "date")
        pd.testing.assert_frame_equal(original, datetime_df)

    def test_extractseasonality_basic(self, datetime_df):
        result = _extract_seasonality(datetime_df, "date")
        assert "date_quarter" in result.columns
        assert "date_is_weekend" in result.columns

    def test_extractseasonality_logs_to_audit(self, datetime_df, audit_trail):
        result = _extract_seasonality(datetime_df, "date", audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["column"] == "date"
        assert audit_trail.trail[0]["step"] == "seasonality"
 

class TestExtractDatetimeFeatures:
    def test_extract_year(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", ["year"])
        assert "date_year" in result.columns
        assert result["date_year"][0] == 2024

    def test_extract_month(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", ["month"])
        assert "date_month" in result.columns
        assert result["date_month"][0] == 1

    def test_extract_day(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", ["day"])
        assert "date_day" in result.columns
        assert result["date_day"][0] == 1

    def test_extract_dayoftheweek(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", ["dayofweek"])
        assert "date_dayofweek" in result.columns
        assert result["date_dayofweek"].iloc[0] == 0 # this should be a monday

    def test_extract_hour(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", ["hour"])
        assert "date_hour" in result.columns

    def test_extract_seasonality_adds_both_columns(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", ["quarter"])
        assert "date_quarter" in result.columns
        assert "date_is_weekend" in result.columns

    def test_extract_lag(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", ["lag"], lag_periods=[1,2,3])
        assert "date_lag_1" in result.columns
        assert "date_lag_2" in result.columns
        assert "date_lag_3" in result.columns

    def test_extract_na(self, datetime_df, caplog):
        import logging
        result = extract_datetime_features(datetime_df, "date", ["unknown_value"])
        with caplog.at_level(logging.WARNING, 
                         logger="toolbox.feature_engineering.datetime"):
            extract_datetime_features(datetime_df, "date", 
                                    features=["unknown_value"])
        assert "unknown_value" in caplog.text

    def test_converts_string_dates(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [10, 20, 30]
        })
        result = extract_datetime_features(df, "date", features=["year"])
        assert "date_year" in result.columns

    def test_none_features_extracts_all(self, datetime_df):
        result = extract_datetime_features(datetime_df, "date", features=None)
        assert "date_year" in result.columns
        assert "date_month" in result.columns
        assert "date_day" in result.columns
        assert "date_dayofweek" in result.columns
        assert "date_hour" in result.columns

    def test_dataframe_unchanged(self, datetime_df):
        original = datetime_df.copy()
        extract_datetime_features(datetime_df, "date", features=["year"])
        pd.testing.assert_frame_equal(datetime_df, original)

    def test_logs_to_audit(self, datetime_df, audit_trail):
        extract_datetime_features(datetime_df, "date", 
                                  features=["year"], audit=audit_trail)
        assert len(audit_trail) >= 1
        assert audit_trail.trail[0]["step"] == "datetime_extraction"
        assert audit_trail.trail[0]["column"] == "date"