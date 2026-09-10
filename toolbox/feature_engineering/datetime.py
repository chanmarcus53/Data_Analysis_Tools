from toolbox.logger import get_logger
from datetime import datetime
import pandas as pd

logger = get_logger(__name__)



def extract_datetime_features(df, column, features=None, audit=None, date_format=None, **kwargs):
    """
    Entry point — extracts requested features from a datetime column.
    
    features: list of features to extract, or None for all
    Available: "year", "month", "day", "dayofweek", 
               "quarter", "is_weekend", "hour", "lag"
    
    example:
        extract_datetime_features(df, "date", features=["month", "dayofweek", "is_weekend"])
        extract_datetime_features(df, "date", features=["lag"], lag_periods=[1, 7, 30])

    date_format: optional strftime format string e.g. "%d/%m/%Y"
                 if None, pandas will try to infer it automatically
    """
    df = df.copy()
    
    # convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        logger.info(f"Converting '{column}' to datetime")
        try:
            df[column] = pd.to_datetime(df[column], format=date_format)
        except Exception as e:
            raise ValueError(f"Could not convert '{column}' to datetime: {e}")

    if features is None:
        features = ["year", "month", "day", "dayofweek", "quarter", "is_weekend", "hour"]

    for feature in features:
        if feature in ["year", "month", "day"]:
            df = _extract_basic(df, column, feature)
        elif feature == "dayofweek":
            df[f"{column}_dayofweek"] = df[column].dt.dayofweek
        elif feature == "hour":
            df[f"{column}_hour"] = df[column].dt.hour
        elif feature in ["quarter", "is_weekend"]:
            df = _extract_seasonality(df, column, audit)
        elif feature == "lag":
            lag_periods = kwargs.get("lag_periods", [1])
            df = _extract_lag(df, column, lag_periods, audit)
        else:
            logger.warning(f"Unknown feature '{feature}' — skipping")

    return df

def _extract_basic(df, column, feature):
    # extracts a single basic feature — year, month, day etc.
    df.copy()
    if feature == "year":
        df[f"{column}_year"] = df[column].dt.year
    elif feature == "month":
        df[f"{column}_month"] = df[column].dt.month
    elif feature == "day":
        df[f"{column}_day"] = df[column].dt.day
    else:
        raise ValueError(f"Unknown basic feature: {feature}, choose from: year, month, day")
    return df

def _extract_lag(df, column, lag_periods, audit=None):
    # creates lag features — value from n periods ago
    # hint: look into series.shift()
    df = df.copy()
    for lag in lag_periods:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        logger.debug(f"Created lag feature '{column}_lag_{lag}'")
    if audit is not None:
        audit.log("lag", column, f"Created lag features for periods: {lag_periods}")
    return df

def _extract_seasonality(df, column, audit=None):
    # creates features for seasonality — quarter, is_weekend etc.
    df = df.copy()
    df[f"{column}_quarter"] = df[column].dt.quarter
    df[f"{column}_is_weekend"] = df[column].dt.dayofweek >= 5
    logger.debug(f"Extracted seasonality features from '{column}'")
    if audit is not None:
        audit.log("seasonality", column, "Extracted quarter and is_weekend")
    return df