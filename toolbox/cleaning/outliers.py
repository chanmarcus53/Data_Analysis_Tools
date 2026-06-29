from toolbox.logger import get_logger
from toolbox.cleaning.audit import AuditTrail
from scipy import stats
import pandas as pd

logger = get_logger(__name__)

def handle_outliers(df, column, method, action, audit=None, **kwargs):
    if audit is None:
        audit = AuditTrail()

    methods = ["iqr", "zscore"]
    actions = ["flag", "cap", "remove"]

    if method not in methods:
        raise ValueError(f"Unsupported method: '{method}'. Choose from: {methods}")
    if action not in actions:
        raise ValueError(f"Unsupported action: '{action}'. Choose from: {actions}")

    mask = _detect(df, column, method, **kwargs)

    if action == "flag":
        return _flag(df, column, mask, audit)
    elif action == "cap":
        return _cap(df, column, mask, method, audit, **kwargs)
    elif action == "remove":
        return _remove(df, column, mask, audit)


def _detect(df, column, method, **kwargs):
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)

    elif method == "zscore":
        threshold = kwargs.get("threshold", 3)
        z_scores = df[column].copy()
        z_scores[df[column].notna()] = stats.zscore(df[column].dropna())
        return z_scores.abs() > threshold


def _flag(df, column, mask, audit=None):
    df = df.copy()
    df[f"{column}_outlier"] = mask
    if audit:
        audit.log("flag", column, f"Flagged {mask.sum()} outliers")
    return df


def _cap(df, column, mask, method, audit=None, **kwargs):
    df = df.copy()

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == "zscore":
        threshold = kwargs.get("threshold", 3)
        lower_bound = df[column].mean() - threshold * df[column].std()
        upper_bound = df[column].mean() + threshold * df[column].std()

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    if audit:
        audit.log("cap", column, f"Capped {mask.sum()} outliers using {method} "
                                  f"bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df


def _remove(df, column, mask, audit=None):
    df = df.copy()
    removed_count = mask.sum()
    df = df[~mask]
    if audit:
        audit.log("remove", column, f"Removed {removed_count} outlier rows")
    return df