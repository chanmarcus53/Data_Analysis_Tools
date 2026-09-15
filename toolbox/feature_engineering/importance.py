from toolbox.logger import get_logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd


logger = get_logger(__name__)

def rank_features(df, target, method="auto", audit=None):
    """
    Entry point — ranks features by importance.
    
    method: "auto", "random_forest", "correlation", "mutual_information"
    
    "auto" detects whether target is continuous or categorical
    and picks the most appropriate method.
    
    Returns a DataFrame sorted by importance score descending.
    
    example:
        rank_features(df, target="price", method="auto")
        rank_features(df, target="status", method="mutual_information")
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    task = _detect_task(df[target])

    if method =="auto":
        if task == "classification":
            method = "mutual_information"
        else:
            method = "random_forest"

    if method == "random_forest":
        result = _random_forest_importance(df, target, task)
    elif method == "correlation":
        result = _correlation_importance(df, target)
    elif method == "mutual_information":
        result = _mutual_information(df, target, task)
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from: auto, random_forest, correlation, mutual_information")

    logger.info(f"Feature ranking complete — top feature: {result['feature'].iloc[0]}")

    if audit is not None:
        audit.log("rank_features", target, f"Ranked {len(result)} features using {method}")

    return result


def _detect_task(series):
    """
    Detects whether the target is classification or regression.
    hint: if the target is numeric with many unique values → regression
          if categorical or few unique values → classification
    think about what threshold of unique values makes sense
    """
    if series.dtype == "object" or series.dtype.name == "category":
        return "classification"
    if series.dtype.kind in "biufc":
        return "classification" if series.nunique() < 20 else "regression"
    return "classification"  # default fallback

    
def _random_forest_importance(df, target, task):
    df = df.copy()
    X = df.drop(columns=[target]).select_dtypes(include="number")
    y = df[target]

    if X.empty:
        raise ValueError("No numeric features found for random forest importance")

    if task == "classification":
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        forest = RandomForestRegressor(n_estimators=100, random_state=42)

    forest.fit(X, y, n_jobs=2)
    logger.debug(f"Random forest fitted with {X.shape[1]} features")

    importance_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": forest.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df

def _correlation_importance(df, target):
    # uses absolute correlation with target as importance score
    # good for quick linear relationships
    """
    Uses absolute correlation with target as importance score.
    Only works for numeric features.
    Returns a DataFrame with columns: feature, importance
    hint: df.corrwith()
    """
    df = df.copy()
    X = df.drop(columns=[target]).select_dtypes(include="number")
    y = df[target]

    if X.empty:
        raise ValueError("No numeric features found for correlation importance")
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    importance_df = pd.DataFrame({
        "feature": corr.index,
        "importance": corr.values
    }).reset_index(drop=True)

    return importance_df


def _mutual_information(df, target, task):
    # uses sklearn mutual_info_regression or mutual_info_classif
    # captures non-linear relationships too
    df = df.copy()
    X = df.drop(columns=[target]).select_dtypes(include="number")
    y = df[target]

    if X.empty:
        raise ValueError("No numeric features found for mutual information importance")
    
    if task == "regression":
        mi = mutual_info_regression(X, y, n_jobs=2)
    else:
        mi = mutual_info_classif(X, y, n_jobs=2)

    importance_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": mi
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df