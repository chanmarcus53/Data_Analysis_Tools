from toolbox.exploratory.stats import summarise, _skewness_kurtosis, _correlation_matrix, _value_counts, _percentiles
import pandas as pd
import pytest

class TestSkewnessKurtosis:
    def test_returns_two_dicts(self, sample_df):
        skewness, kurtosis = _skewness_kurtosis(sample_df)
        assert isinstance(skewness, dict)
        assert isinstance(kurtosis, dict)

    def test_only_numeric_columns(self, sample_df):
        skewness, kurtosis = _skewness_kurtosis(sample_df)
        assert "name" not in skewness
        assert "school" not in skewness

    def test_correct_columns_present(self, sample_df):
        skewness, kurtosis = _skewness_kurtosis(sample_df)
        assert "grade" in skewness
        assert "grade" in kurtosis

    def test_values_are_rounded(self, sample_df):
        skewness, kurtosis = _skewness_kurtosis(sample_df)
        for val in skewness.values():
            assert val == round(val, 4)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "status": ["active", "inactive"]})
        skewness, kurtosis = _skewness_kurtosis(df)
        assert skewness == {}
        assert kurtosis == {}

class TestCorrelationMatrix:
    def test_pearson_returns_dict(self, sample_df):
        result = _correlation_matrix(sample_df, "pearson")
        assert isinstance(result, dict)

    def test_spearman_returns_dict(self, sample_df):
        result = _correlation_matrix(sample_df, "spearman")
        assert isinstance(result, dict)

    def test_kendall_returns_dict(self, sample_df):
        result = _correlation_matrix(sample_df, "kendall")
        assert isinstance(result, dict)

    def test_invalid_method_raises(self, sample_df):
        with pytest.raises(ValueError):
            _correlation_matrix(sample_df, "unsupported")

    def test_diagonal_is_one(self, sample_df):
        result = _correlation_matrix(sample_df, "pearson")
        for col in result:
            assert result[col][col] == pytest.approx(1.0)

class TestValueCounts:
    def test_returns_dict(self, sample_df):
        result = _value_counts(sample_df)
        assert isinstance(result, dict)

    def test_only_categorical_columns(self, sample_df):
        result = _value_counts(sample_df)
        assert "grade" not in result
        assert "salary" not in result

    def test_correct_columns_present(self, sample_df):
        result = _value_counts(sample_df)
        assert "school" in result
        assert "name" in result

    def test_counts_are_correct(self, sample_df):
        result = _value_counts(sample_df)
        assert result["school"]["A"] == 3
        assert result["school"]["C"] == 4

    def test_no_categorical_columns(self):
        df = pd.DataFrame({"grade": [50, 60, 70], "salary": [30000, 40000, 50000]})
        result = _value_counts(df)
        assert result == {}

class TestPercentiles:
    def test_returns_dict(self, sample_df):
        result = _percentiles(sample_df)
        assert isinstance(result, dict)

    def test_correct_percentile_keys(self, sample_df):
        result = _percentiles(sample_df)
        for col in result:
            assert 0.05 in result[col]
            assert 0.5 in result[col]

    def test_only_numeric_columns(self, sample_df):
        result = _percentiles(sample_df)
        assert "name" not in result
        assert "school" not in result

    def test_median_correct(self, sample_df):
        result = _percentiles(sample_df)
        expected_median = sample_df["grade"].median()
        assert result["grade"][0.5] == pytest.approx(expected_median, rel=1e-3)

class TestSummarise:
    def test_returns_dict(self, sample_df):
        result = summarise(sample_df)
        assert isinstance(result, dict)

    def test_has_all_keys(self, sample_df):
        result = summarise(sample_df)
        assert "skewness" in result
        assert "kurtosis" in result
        assert "correlations" in result
        assert "value_counts" in result
        assert "percentiles" in result

    def test_correlations_has_all_methods(self, sample_df):
        result = summarise(sample_df)
        assert "pearson" in result["correlations"]
        assert "spearman" in result["correlations"]
        assert "kendall" in result["correlations"]