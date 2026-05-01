"""
This module provides a unified interface for loading data from various sources into pandas DataFrames. It supports:
- File-based sources: CSV, Excel, JSON, Parquet
- SQL databases: PostgreSQL, MySQL, SQLite, MSSQL
- REST APIs: Fetching JSON data and converting to DataFrame
The main function is `load()`, which detects the source type and dispatches to the appropriate
loader function. The module also includes error handling for unsupported formats and edge cases in API responses.
Last updated: 2026-04-30
By: Marcus Chan
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
import requests
from toolbox.logger import get_logger

# Global logger for this module
logger = get_logger(__name__)
# Global constants for common keys in API responses and supported file extensions
COMMON_KEYS = ["results", "data", "items", "records", "rows", "content"]
# Global constant for supported file extensions and their corresponding pandas readers
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".json", ".parquet"]

def load(source, **kwargs):
    """
    Unified entry point. Detects source type and dispatches accordingly.
    Returns a pandas DataFrame.
    """
    if isinstance(source, str) and source.startswith("http"):
        return _load_api(source, **kwargs)
    elif _is_sql_connection(source):
        return _load_sql(source, **kwargs)
    else:
        return _load_file(source, **kwargs)


def _load_file(path, **kwargs):
    """
    Detect file extension and call the appropriate reader.
    Raise a clear error if the extension isn't supported.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are: {SUPPORTED_EXTENSIONS}")
    
    parser_map = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet
    }

    reader = parser_map[ext]
    return reader(path, **kwargs)


def _load_sql(connection, query, **kwargs):
    """
    Execute a SQL query against a connection and return results.
    hint: look into pd.read_sql() and SQLAlchemy engine strings
    """
    database_index = connection.lower().find("://")
    if database_index == -1:
        raise ValueError("Invalid SQL connection string. Unable to determine database type")
    elif database_index < 5:
        raise ValueError("Invalid SQL connection string. Database type appears too short to be valid")
    else:
        database_type = connection[:database_index].lower()
        if database_type not in ["postgresql", "mysql", "sqlite", "mssql", "mysql+pymysql", "mssql+pyodbc", 
                                 "postgresql+psycopg2", "sqlite+pysqlite", "oracle+cx_oracle"]:
            raise ValueError(f"Unsupported database type: {database_type}. Supported types are: postgresql, mysql, sqlite, mssql")
    
    engine = create_engine(connection)
    return pd.read_sql(query, con=engine, **kwargs)


def _find_records(data):
    for key in COMMON_KEYS:
        if key in data:
            value = data[key]
            
            # edge case 1: key exists but value isn't a list
            if not isinstance(value, list):
                logger.warning(
                    f"Found key '{key}' but value is {type(value).__name__}, expected a list. Wrapping in a list."
                )
                return [value]
            
            return value
    
    # edge case 2: no common key matched
    logger.warning(
        f"Could not find records in response keys: {list(data.keys())}. "
        "Add the correct key to COMMON_KEYS if needed."
    )
    return None

def _load_api(url, params=None, headers=None, **kwargs):
    """
    Fetch JSON data from a REST endpoint and return as a DataFrame.
    hint: look into the requests library and how to handle pagination
    """
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = _find_records(data)
        if records is None:
            logger.warning("Returning empty DataFrame due to unresolved response structure.")
            return pd.DataFrame()
    else:
        raise ValueError(f"Unexpected response type: {type(data)}")

    return pd.DataFrame(records)


def _is_sql_connection(source):
    """
    Return True if source looks like a SQL connection string or SQLAlchemy engine.
    hint: what types or string patterns would indicate a DB connection?
    """
    raise NotImplementedError