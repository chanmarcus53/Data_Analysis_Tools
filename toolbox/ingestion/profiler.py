from toolbox.logger import get_logger

logger = get_logger(__name__)

def profile(df):
    """
    Return a dict summarising the DataFrame.
    """
    logger.info(f"Starting profile: {df.shape[0]} rows, {df.shape[1]} columns")
    column_profiles = _profile_columns(df)
    result = {
        "shape": _get_shape(df),
        "memory": _get_memory(df),
        "columns": column_profiles,
        "warnings": _detect_issues(df, column_profiles)
    }
    logger.info(f"Profile complete. Memory usage: {result['memory']}MB")
    return result


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
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(include=['object', 'category'])
    
    return_dict = {}
    for col_name in numeric_df.columns():
        return_dict[col_name] = _profile_numeric(numeric_df[col_name])

    for col_name in categorical_df.columns():
        return_dict[col_name] = _profile_categorical(categorical_df[col_name])

    logger.debug(f"Profiled {len(df.columns)} columns")
    return return_dict

def _profile_numeric(series):
    """
    Returns a dictionary of an overview of the numerical column

    Parameters:
    - series: Pandas series from a part of a dataframe

    Returns:
    - dict: series stats
    """
    return {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "null_count": int(series.isnull().sum()),
        "null_pct": round(series.isnull().mean() * 100, 2),
        "mean": round(series.mean(), 4),
        "std": round(series.std(), 4),
        "min": round(series.min(), 4),
        "max": round(series.max(), 4),
        "skewness": round(series.skew(), 4)
    }

def _profile_categorical(series):
    return {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "null_count": int(series.isnull().sum()),
        "null_pct": round(series.isnull().mean() * 100, 2),
        "unique_count": int(series.nunique()),
        "top_values": series.value_counts().head(5).to_dict() 
    }

def _detect_issues(df, column_profile): 
    """
    Return a list of warning strings for common data quality problems.
    Things to flag: high null %, single-value columns, likely duplicates
    """
    issue_list = []
    for key, value in column_profile.items():
        if column_profile[key]['null_pct'] > 5:
            msg = f"""The column {key} has a high percentage of null values 
                    {column_profile[key]['null_pct']}, with a count:{column_profile[key]['null_count']}"""
            issue_list.append(msg)
            logger.warning(msg)
            
        # only checking the outliers for numeric columns
        if "mean" in value:
            if value['mean'] + 2 * value['std'] < value['max']:
                msg = f"""The column: {key} has a potential upper outlier"""
                issue_list.append(msg)
                logger.warning(msg)
            if value['mean'] - 2 * value['std'] > value['min']:
                msg = f"""The column: {key} has a potential lower outlier"""
                issue_list.append(msg)
                logger.warning(msg)
        
        if value.get("unique_count") == 1:
            issue_list.append(f"Column '{key}' has only one unique value and may be uninformative")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        msg = f"{duplicate_count} duplicated rows detected"
        issue_list.append(msg)
        logger.warning(msg)
