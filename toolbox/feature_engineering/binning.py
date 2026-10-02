from toolbox.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def bin(df, column, method, n_bins=1, audit=None, **kwargs):
    """
    Creates bins the covers a range of values to make columns easier to group

    Input
    df - Dataframe
    column - column used for bin range
    method - [equal_width, equal_frequency, custom_bins]
        - when using custom bins include:
            - "bins" input eg. [0, 18, 65, 100] # as their name
    n_bins - number of bins to divide the dataframe into
    audit - audit trail

    Optional:
    bins - used for custom bins method
    """
    logger.info(f"Binning column '{column}' using method '{method}'")
    # entry point — dispatches to binning strategy
    if method == "equal_width":
        return _equal_width(df, column, n_bins, audit=audit, **kwargs)
    elif method == "equal_frequency":
        return _equal_frequency(df, column, n_bins, audit=audit, **kwargs)
    elif method == "custom_bins":
        bins = kwargs.pop("bins", None)
        return _custom_bins(df, column, bins=bins, audit=audit, **kwargs)
    else:
        raise ValueError(f"Unknown binning method: {method}")


def _equal_width(df, column, n_bins, labels=None, audit=None):
    """
    Divides the dataframe into equal width bins

    Input
    df - dataframe
    column - column used for binning
    n_bins - number of bins
    labels - setting bin names
    audit - audit trail
    """
    # divide range into equal width bins
    df = df.copy()
    df[f"{column}_binned"] = pd.cut(df[column], bins=n_bins, labels=labels)
    logger.debug(f"Equal width binning: {n_bins} bins on '{column}'")
    if audit is not None:
        audit.log("equal_width", column, f"Binned info {n_bins} equal width bins")
    return df

def _equal_frequency(df, column, n_bins, labels=None, audit=None):
    """
    Divided the dataframe such that each bin equalizes the number of rows

    Input
    df - Dataframe
    column - column used for binning
    n_bins - number of bins
    labels - setting bin names
    audit - audit trail
    """
    # divide into bins with equal number of records
    df = df.copy()
    try:
        df[f"{column}_binned"] = pd.qcut(df[column], q=n_bins, labels=labels)
    except ValueError:
        logger.warning(f"Duplicate values detected in '{column}', dropping duplicate bin edges")
        df[f"{column}_binned"] = pd.qcut(df[column], q=n_bins, labels=labels, duplicates="drop")
    
    if audit is not None:
        audit.log("equal_frequency", column, f"Binned into {n_bins} equal frequency bins")
    return df

def _custom_bins(df, column, bins, labels=None, audit=None):
    """
    Setting up custom bins in the dataframe, binned based on closest value

    Input
    df - Dataframe
    column - column used for binning
    bins - list of values for the center of bins
    labels - setting bin names
    audit - audit trail
    """
    # user defined bin boundaries
    if bins is None:
        raise ValueError("bins parameter must be provided for custom binning")
    df = df.copy()
    df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
    logger.debug(f"Custom binning on '{column}' with {len(bins) - 1} bins")
    if audit is not None:
        audit.log("custom_bins", column, f"Binned into {len(bins) - 1} custom bins")
    return df