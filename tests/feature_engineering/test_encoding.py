from toolbox.feature_engineering.encoding import encode, _ordinal_encode, _onehot_encode, _target_encode
import pytest
import pandas as pd

class TestOrdinalEncode:
    def test_dataframe_unchanged(self, encoding_df):
        original = encoding_df.copy()
        _ordinal_encode(encoding_df, "size", order=["small", "medium", "large"])
        pd.testing.assert_frame_equal(encoding_df, original)

    def test_ordinal_encode_with_order(self, encoding_df):
        order = ["small", "medium", "large"]
        result = _ordinal_encode(encoding_df, "size", order=order)
        assert "size_encoded" in result.columns
        assert result["size_encoded"].tolist() == [0, 1, 2, 0, 2, 1]

    def test_ordinal_encode_without_order_raises(self, encoding_df):
        with pytest.raises(ValueError):
            _ordinal_encode(encoding_df, "size")

    def test_unknown_category_encoded_as_minus_one(self, encoding_df):
        result = _ordinal_encode(encoding_df, "size", order=["small", "medium"])
        # "large" is not in order so should be -1
        assert -1 in result["size_encoded"].values

    def test_logs_to_audit(self, encoding_df, audit_trail):
        _ordinal_encode(encoding_df, "size", order=["small", "medium", "large"], audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["column"] == "size"

class TestOneHotEncode:
    def test_dataframe_unchanged(self, encoding_df):
        original = encoding_df.copy()
        _onehot_encode(encoding_df, "color")
        pd.testing.assert_frame_equal(encoding_df, original)

    def test_onehot_encode_default(self, encoding_df):
        result = _onehot_encode(encoding_df, "color")
        assert "color_red" in result.columns
        assert "color_blue" in result.columns
        assert "color_green" in result.columns

    def test_onehot_encode_drop_first(self, encoding_df):
        result = _onehot_encode(encoding_df, "color", drop_first=True)
        assert "color_red" not in result.columns
        assert "color_blue" in result.columns
        assert "color_green" in result.columns

    def test_onehot_keeps_original_when_drop_original_false(self, encoding_df):
        result = _onehot_encode(encoding_df, "color", drop_original=False)
        assert "color" in result.columns
        assert "color_red" in result.columns

    def test_onehot_drops_original_by_default(self, encoding_df):
        result = _onehot_encode(encoding_df, "color")
        assert "color" not in result.columns

    def test_logs_to_audit(self, encoding_df, audit_trail):
        _onehot_encode(encoding_df, "color", audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "onehot_encode"
        assert audit_trail.trail[0]["column"] == "color"


class TestTargetEncode:
    def test_dataframe_unchanged(self, encoding_df):
        result = _target_encode(encoding_df, "size", target="price")
        assert not result.equals(encoding_df)
        assert "size_encoded" in result.columns

    def test_target_encode_with_target(self, encoding_df):
        result = _target_encode(encoding_df, "size", target="price")
        assert "size_encoded" in result.columns
        expected_means = encoding_df.groupby("size")["price"].mean().to_dict()
        for size, mean in expected_means.items():
            assert result[result["size"] == size]["size_encoded"].iloc[0] == mean

    def test_target_encode_without_target_raises(self, encoding_df):
        with pytest.raises(ValueError):
            _target_encode(encoding_df, "size", target=None)

    def test_logs_to_audit(self, encoding_df, audit_trail):
        _target_encode(encoding_df, "size", audit=audit_trail)
        assert len(audit_trail) == 1
        assert audit_trail.trail[0]["step"] == "target_encode"
        assert audit_trail.trail[0]["column"] == "size"
    

class TestEncode:
    def test_encode_ordinal(self, encoding_df):
        result = encode(encoding_df, "size", method="ordinal", 
                       order=["small", "medium", "large"])
        assert "size_encoded" in result.columns

    def test_encode_onehot(self, encoding_df):
        result = encode(encoding_df, "color", method="onehot")
        assert "color_red" in result.columns

    def test_encode_target(self, encoding_df):
        result = encode(encoding_df, "size", method="target", target="price")
        assert "size_encoded" in result.columns

    def test_unsupported_method_raises(self, encoding_df):
        with pytest.raises(ValueError):
            encode(encoding_df, "size", method="unsupported")