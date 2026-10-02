from toolbox.logger import get_logger
from toolbox.cleaning.audit import AuditTrail
from scipy import stats
import pandas as pd

logger = get_logger(__name__)

def handle_outliers(df, column, method, action, audit=None, **kwargs):
    """
    Handles outliers and how to handle them through just flagging them, capping the values, or removing them

    Input
    df - the dataframe we are operating on
    column - the column in the dataframe we are operative on
    method - How to determine the outlier [iqr, z-score]
    actions - How to handle outlier [flag, cap, remove]
    audit - Audit Trail for logging
    """
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
    """
    Detects Outliers via either z-score method or IQR method

    Input
    df - the dataframe we are operating on
    column - the column in the dataframe we are operative on
    method - [z-score, IQR]
        - threshold - used for z-score to determine cutoff
    """
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == "zscore":
        threshold = kwargs.get("threshold", 3)
        z_scores = df[column].astype(float).copy()
        z_scores[df[column].notna()] = stats.zscore(df[column].dropna())
        return z_scores.abs() > threshold
    else:
        raise ValueError(f"Unsupported method: '{method}'")
    


def _flag(df, column, mask, audit=None):
    """
    Flags outliers identified in the mask (gained from the _detect function)

    Input
    df - the dataframe we are operating on
    column - the column in the dataframe we are operative on
    mask - determinator of outliers from _detect function
    """
    df = df.copy()
    df[f"{column}_outlier"] = mask
    if audit is not None:
        audit.log("flag", column, f"Flagged {mask.sum()} outliers")
    return df


def _cap(df, column, mask, method, audit=None, **kwargs):
    """
    Caps outliers to an upper and lower bound, identified in the mask (gained from the _detect function)

    Input
    df - the dataframe we are operating on
    column - the column in the dataframe we are operative on
    mask - determinator of outliers from _detect function
    """
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
    if audit is not None:
        audit.log("cap", column, f"Capped {mask.sum()} outliers using {method} "
                                  f"bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df


def _remove(df, column, mask, audit=None):
    """
    Deletes outliers identified in the mask (gained from the _detect function)

    Input
    df - the dataframe we are operating on
    column - the column in the dataframe we are operative on
    mask - determinator of outliers from _detect function
    """
    df = df.copy()
    removed_count = mask.sum()
    df = df[~mask]
    if audit is not None:
        audit.log("remove", column, f"Removed {removed_count} outlier rows")
    return df