from toolbox.exploratory.comparison import compare, _compare_groups, _compare_periods, _test_significance, _check_normality
import pandas as pd
import pytest

class TestCheckNormality:
    def test_returns_bool(self, sample_df):
        result = _check_normality(sample_df["grade"].dropna())
        assert isinstance(result, bool)

    def test_small_sample_returns_false(self):
        series = pd.Series([1, 2])
        result = _check_normality(series)
        assert result == False

    def test_handles_nulls(self, sample_df):
        result = _check_normality(sample_df["grade"])
        assert isinstance(result, bool)

class TestTestSignificance:
    def test_two_groups_returns_dict(self, sample_df):
        group_a = sample_df[sample_df["school"] == "A"]["grade"].dropna()
        group_b = sample_df[sample_df["school"] == "B"]["grade"].dropna()
        result = _test_significance(group_a, group_b)
        assert isinstance(result, dict)

    def test_has_all_keys(self, sample_df):
        group_a = sample_df[sample_df["school"] == "A"]["grade"].dropna()
        group_b = sample_df[sample_df["school"] == "B"]["grade"].dropna()
        result = _test_significance(group_a, group_b)
        assert "test_used" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "interpretation" in result

    def test_significant_is_bool(self, sample_df):
        group_a = sample_df[sample_df["school"] == "A"]["grade"].dropna()
        group_b = sample_df[sample_df["school"] == "B"]["grade"].dropna()
        result = _test_significance(group_a, group_b)
        assert isinstance(result["significant"], bool)

    def test_three_groups(self, categorical_df):
        groups = [
            categorical_df[categorical_df["school"] == s]["grade"].dropna()
            for s in ["A", "B", "C"]
        ]
        result = _test_significance(*groups)
        assert result["test_used"] in ["anova", "kruskal-wallis"]

    def test_single_group_raises(self, sample_df):
        group = sample_df["grade"].dropna()
        with pytest.raises(ValueError):
            _test_significance(group)

class TestCompareGroups:
    def test_returns_dict(self, categorical_df):
        result = _compare_groups(categorical_df, "grade", "school")
        assert isinstance(result, dict)

    def test_has_all_keys(self, categorical_df):
        result = _compare_groups(categorical_df, "grade", "school")
        assert "column" in result
        assert "by" in result
        assert "groups" in result
        assert "significance" in result

    def test_group_stats_correct_keys(self, categorical_df):
        result = _compare_groups(categorical_df, "grade", "school")
        for group in result["groups"].values():
            assert "count" in group
            assert "mean" in group
            assert "std" in group
            assert "median" in group

    def test_all_groups_present(self, categorical_df):
        result = _compare_groups(categorical_df, "grade", "school")
        assert "A" in result["groups"]
        assert "B" in result["groups"]
        assert "C" in result["groups"]

class TestComparePeriods:
    def test_returns_dict(self, time_series_df):
        result = _compare_periods(time_series_df, "grade", "quarter", "Q1", "Q2")
        assert isinstance(result, dict)

    def test_has_all_keys(self, time_series_df):
        result = _compare_periods(time_series_df, "grade", "quarter", "Q1", "Q2")
        assert "column" in result
        assert "period_col" in result
        assert "periods" in result
        assert "significance" in result

    def test_invalid_period_raises(self, time_series_df):
        with pytest.raises(ValueError):
            _compare_periods(time_series_df, "grade", "quarter", "Q1", "Q5")

    def test_period_stats_correct(self, time_series_df):
        result = _compare_periods(time_series_df, "grade", "quarter", "Q1", "Q2")
        assert result["periods"]["Q1"]["count"] == 3
        assert result["periods"]["Q2"]["count"] == 3

class TestCompare:
    def test_group_comparison(self, categorical_df):
        result = compare(categorical_df, "grade", by="school")
        assert "groups" in result

    def test_period_comparison(self, time_series_df):
        result = compare(time_series_df, "grade",
                        period_col="quarter", period_a="Q1", period_b="Q2")
        assert "periods" in result

    def test_both_raises(self, categorical_df):
        with pytest.raises(ValueError):
            compare(categorical_df, "grade", by="school",
                   period_col="quarter", period_a="Q1", period_b="Q2")

    def test_neither_raises(self, categorical_df):
        with pytest.raises(ValueError):
            compare(categorical_df, "grade")