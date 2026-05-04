from toolbox.logger import get_logger

logger = get_logger(__name__)

def profile(df):
    """
    Return a dict summarising the DataFrame.
    """
    return {
        "shape": _get_shape(df),
        "memory": _get_memory(df),
        "columns": _profile_columns(df),
        "warnings": _detect_issues(df)
    }


def _get_shape(df):
    """
    Return the row count and columns count

    Parameters:
    - df : Pandas Dataframe

    Returns:
    - df.shape[0]: row count
    - df.shape[1]: column count
    """
    return df.shape[0], df.shape[1]

def _get_memory(df):
    """
    Returns memory usage of the dataframe in megabytes

    Parameters:
    - df: Pandas Dataframe

    Returns:
    - memory: Floating point, 3 decimal places
    """
    return round(df.memory_usage(index=True, deep=True).sum()/1_000_000, 3)

def _profile_columns(df):
    """
    Return a per-column summary. Numeric and categorical columns
    need different treatment.
    hint: df.select_dtypes() will be useful here
    """
    # TODO: iterate columns, detect type, call the right sub-profiler
    raise NotImplementedError

def _profile_numeric(series):
    # TODO: return count, nulls, mean, std, min, max, skewness
    # hint: pandas has most of these built in — what about skewness?
    raise NotImplementedError

def _profile_categorical(series):
    # TODO: return count, nulls, unique count, top 5 values + frequencies
    raise NotImplementedError

def _detect_issues(df):
    """
    Return a list of warning strings for common data quality problems.
    Things to flag: high null %, single-value columns, likely duplicates
    """
    # TODO: define your own thresholds — what % nulls counts as 'high'?
    raise NotImplementedError