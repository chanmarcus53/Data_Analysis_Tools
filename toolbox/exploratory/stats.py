from toolbox.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def summarise(df):
    logger.info(f"Running stats summary on DataFrame with shape {df.shape}")
    skewness, kurtosis = _skewness_kurtosis(df)
    result = {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "correlations": {
            "pearson": _correlation_matrix(df, "pearson"),
            "spearman": _correlation_matrix(df, "spearman"),
            "kendall": _correlation_matrix(df, "kendall")
        },
        "value_counts": _value_counts(df),
        "percentiles": _percentiles(df)
    }
    logger.info("Stats summary complete")
    return result

def _skewness_kurtosis(df):
    numeric_df = df.select_dtypes(include="number")
    skewness = {}
    kurtosis = {}
    for col in numeric_df.columns:
        skewness[col] = round(numeric_df[col].skew(), 4)
        kurtosis[col] = round(numeric_df[col].kurt(), 4)
    return skewness, kurtosis

def _correlation_matrix(df, method):
    if method.lower() not in ["pearson", "spearman", "kendall"]:
        raise ValueError(f"Invalid method: '{method}'. Choose from 'pearson', 'spearman', 'kendall'.")
    result = df.corr(method=method.lower(), numeric_only=True)
    logger.debug(f"Computed {method} correlation matrix")
    return result.to_dict()

def _value_counts(df):
    categorical_col = df.select_dtypes(include=["object", "category"])
    return {col: categorical_col[col].value_counts().to_dict()
            for col in categorical_col.columns}

def _percentiles(df):
    percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    result = df.quantile(percentiles, numeric_only=True).transpose()
    return result.to_dict()