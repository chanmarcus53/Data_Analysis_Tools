from toolbox.feature_engineering.binning import bin, _equal_width, _equal_frequency, _custom_bins
import pytest
import pandas as pd

class TestEqualWidth:
    def test_dataframe_unchanged(self, binning_df):
        original = binning_df.copy()
        result = _equal_width(binning_df, "age", 5)
        pd.testing.assert_frame_equal(original, binning_df)

    def test_equalwidth(self, binning_df):
        labels = ["tiny", "small", "medium", "large", "x-large"]
        result = _equal_width(binning_df, "id", 5, labels=labels)
        assert "id_binned" in result.columns
        assert result["id_binned"].tolist() == ["tiny", "tiny", "small", "small", "medium", "medium", "large", "large", "x-large", "x-large"]

    def test_equalwidth_labelless(self, binning_df):
        result = _equal_width(binning_df, "age", 5)
        assert "age_binned" in result.columns
        # check number of unique bins instead of exact values
        assert result["age_binned"].nunique() == 5
        
    def test_logs_to_audit(self, binning_df, audit_trail):
        labels = ["tiny", "small", "medium", "large", "x-large"]
        result = _equal_width(binning_df, "id", 5, labels=labels, audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "equal_width"
        assert audit_trail.trail[0]["column"] == "id"


class TestEqualFrequency:
    def test_dataframe_unchanged(self, binning_df):
        original = binning_df.copy()
        labels = ["child", "young_adult", "adult", "elder", "senior"]
        result = _equal_frequency(binning_df, "age", 5, labels=labels)
        pd.testing.assert_frame_equal(binning_df, original)

    def test_equalfrequency(self, binning_df):
        labels = ["child", "young_adult", "adult", "elder", "senior"]
        result = _equal_frequency(binning_df, "age", 5, labels=labels)
        assert "age_binned" in result.columns
        assert result["age_binned"] == ["child", "child", "young_adult", "young_adult", "adult", "adult", "elder", "elder", "senior", "senior"]

    def test_equalfrequency_labelless(self, binning_df):
        result = _equal_frequency(binning_df, "age", 5)
        assert "age_binned" in result.columns
        assert result["age_binned"] == [1,1,2,2,3,3,4,4,5,5]

    def test_logs_to_audit(self, binning_df, audit_trail):
        labels = ["tiny", "small", "medium", "large", "x-large"]
        result = _equal_frequency(binning_df, "age", 5, labels=labels, audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "equal_frequency"
        assert audit_trail.trail[0]["column"] == "age"
    
class TestCustomBins:
    def test_dataframe_unchanged(self, binning_df):
        original = binning_df.copy()
        _custom_bins(binning_df, "age", bins=[0, 18, 65, 100])
        pd.testing.assert_frame_equal(binning_df, original)

    def test_custom_bins_correct_columns(self, binning_df):
        result = _custom_bins(binning_df, "age", bins=[0, 18, 65, 100])
        assert "age_binned" in result.columns

    def test_custom_bins_correct_values(self, binning_df):
        # age column: [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        # bins: [0, 18, 65, 100] → child, adult, senior
        labels = ["child", "adult", "senior"]
        result = _custom_bins(binning_df, "age", bins=[0, 18, 65, 100], labels=labels)
        assert result["age_binned"].iloc[0] == "child"   # age 5
        assert result["age_binned"].iloc[2] == "adult"   # age 25
        assert result["age_binned"].iloc[7] == "senior"  # age 75

    def test_custom_bins_none_raises(self, binning_df):
        with pytest.raises(ValueError):
            _custom_bins(binning_df, "age", bins=None)

    def test_custom_bins_logs_to_audit(self, binning_df, audit_trail):
        _custom_bins(binning_df, "age", bins=[0, 18, 65, 100], audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "custom_bins"

class TestBin:
    def test_equalwidth(self, binning_df, audit_trail):
        result = bin(binning_df, "age", "equal_width", n_bins=5, audit=audit_trail)
        assert "age_binned" in result.columns

    def test_equalfrequency(self, binning_df, audit_trail):
        result = bin(binning_df, "age", "equal_frequency", n_bins=5, audit=audit_trail)
        assert "age_binned" in result.columns

    def test_custombins(self, binning_df, audit_trail):
        result = bin(binning_df, "age", "custom_bins", 
                bins=[0, 18, 65, 100], audit=audit_trail)
        assert "age_binned" in result.columns

    def test_unsupported_method(self, binning_df, audit_trail):
        with pytest.raises(ValueError):
            bin(binning_df, "age", "unsupported_method", n_bins=5, audit=audit_trail)