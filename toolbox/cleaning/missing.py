from toolbox.logger import get_logger
from toolbox.cleaning.audit import AuditTrail
from sklearn.impute import KNNImputer
import pandas as pd

logger = get_logger(__name__)

def handle_missing(df, column, strategy, audit=None, **kwargs):
    if audit is None:
        audit = AuditTrail()

    strategies = ["flag", "drop", "mean", "median", "mode", "constant", "knn", "ffill", "bfill"]
    if strategy not in strategies:
        raise ValueError(f"Unsupported strategy: '{strategy}'. Choose from: {strategies}")

    if strategy == "flag":
        return _flag(df, column, audit)
    elif strategy == "drop":
        return _drop(df, column, audit=audit, **kwargs)
    elif strategy in ["mean", "median", "mode", "constant"]:
        return _impute_simple(df, column, strategy, audit=audit, **kwargs)
    elif strategy == "knn":
        return _impute_knn(df, column, audit=audit, **kwargs)
    elif strategy == "ffill":
        return _impute_ffill(df, column, audit)
    elif strategy == "bfill":
        return _impute_bfill(df, column, audit)


def _flag(df, column, audit=None):
    df = df.copy()
    new_col = f"{column}_missing"
    df[new_col] = df[column].isnull()
    if audit is not None:
        null_count = df[column].isnull().sum()
        audit.log("flag", column, f"Added flag column '{new_col}' — {null_count} nulls flagged")
    return df


def _drop(df, column, threshold=None, axis=0, audit=None):
    df = df.copy()
    before = len(df)

    if threshold is not None:
        thresh_count = int((1 - threshold) * (len(df) if axis == 0 else len(df.columns)))
        df = df.dropna(axis=axis, thresh=thresh_count)
    else:
        df = df.dropna(axis=axis, subset=[column] if axis == 0 else None)

    dropped = before - len(df)
    if audit is not None:
        audit.log("drop", column, f"Dropped {dropped} {'rows' if axis == 0 else 'columns'} "
                                   f"with {'>' + str(threshold * 100) + '%' if threshold else 'any'} nulls")
    return df


def _impute_simple(df, column, strategy, value=None, audit=None):
    df = df.copy()
    null_count = df[column].isnull().sum()

    if strategy == "mean":
        impute_value = df[column].mean()
    elif strategy == "median":
        impute_value = df[column].median()
    elif strategy == "mode":
        impute_value = df[column].mode()[0]
    elif strategy == "constant":
        if value is None:
            raise ValueError("Value must be provided for constant imputation")
        impute_value = value

    df[column] = df[column].fillna(impute_value)
    if audit is not None:
        audit.log("impute_simple", column, f"Imputed {null_count} nulls with {strategy} value: {impute_value}")
    return df


def _impute_knn(df, column, n_neighbors=5, audit=None):
    df = df.copy()
    null_count = df[column].isnull().sum()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df[column] = df_imputed[column]
    if audit is not None:
        audit.log("impute_knn", column, f"Imputed {null_count} nulls using KNN with n_neighbors={n_neighbors}")
    return df


def _impute_ffill(df, column, audit=None):
    df = df.copy()
    null_count = df[column].isnull().sum()
    df[column] = df[column].ffill()
    if audit is not None:
        audit.log("impute_ffill", column, f"Imputed {null_count} nulls using forward fill")
    return df


def _impute_bfill(df, column, audit=None):
    df = df.copy()
    null_count = df[column].isnull().sum()
    df[column] = df[column].bfill()
    if audit is not None:
        audit.log("impute_bfill", column, f"Imputed {null_count} nulls using backward fill")
    return df