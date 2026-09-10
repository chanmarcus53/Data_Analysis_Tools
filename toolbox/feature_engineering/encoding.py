from toolbox.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def encode(df, column, method, audit=None, **kwargs):
    if method == "ordinal":
        return _ordinal_encode(df, column, order=kwargs.get("order"), audit=audit)
    elif method == "onehot":
        return _onehot_encode(df, column, 
                              drop_first=kwargs.get("drop_first", False),
                              drop_original=kwargs.get("drop_original", True),
                              audit=audit)
    elif method == "target":
        return _target_encode(df, column, target=kwargs.get("target"), audit=audit)
    else:
        raise ValueError (f"Unknown encoding method: {method} Choose from: ordinal, onehot, target")

def _ordinal_encode(df, column, order=None, audit=None):
    # encode categories as integers in a specified order
    # hint: look into sklearn OrdinalEncoder or pandas Categorical
    if order is None:
        raise ValueError("order parameter must be provided for ordinal encoding")

    
    df = df.copy()
    unknown = set(df[column].dropna().unique()) - set(order)
    if unknown:
        logger.warning(f"Unknown categories {unknown} found in ")

    df[column + "_encoded"] = pd.Categorical(df[column], categories=order, ordered=True).codes
    logger.debug(f"Ordinal encoding on '{column}' with order {order}")
    if audit is not None:
        audit.log("ordinal_encode", column, f"Encoded with order {order}")
    return df

def _onehot_encode(df, column, drop_first=False, drop_original=True, audit=None):
    # create binary columns for each category
    # hint: look into pd.get_dummies()
    df = df.copy()
    df_encoded = pd.get_dummies(df[column], prefix=column, drop_first=drop_first, dtype=int)
    df = pd.concat([df, df_encoded], axis=1)
    if drop_original:
        df = df.drop(columns=[column])
    logger.debug(f"One-hot encoding on '{column}', drop_first={drop_first}, drop_original={drop_original}")
    if audit is not None:
        new_cols = list(df_encoded.columns)
        audit.log("onehot_encode", column, f"Created columns: {new_cols}, drop_first={drop_first}")
    return df

def _target_encode(df, column, target, audit=None):
    # replace category with mean of target variable per category
    # useful for high cardinality columns
    if target is None:
        raise ValueError("target parameter is required for target encoding")
    
    logger.warning("Target encoding computed on full dataset — ensure you fit only on training data to avoid leakage")
    df = df.copy()
    target_means = df.groupby(column)[target].mean()
    df[f"{column}_encoded"] = df[column].map(target_means)
    logger.debug(f"Target encoding on '{column}' with target '{target}'")
    if audit is not None:
        audit.log("target_encode", column, f"Target encoded using '{target}', {len(target_means)} categories mapped")
    return df
