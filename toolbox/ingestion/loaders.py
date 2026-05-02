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
from sqlalchemy import create_engine, text
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


def _load_sql(connection, query, params=None, chunksize=None, **kwargs):
    """
    Execute a SQL query against a connection and return results as a DataFrame.
    
    Args:
        connection: SQLAlchemy connection string
        query:      SQL query string, use :param_name style placeholders
        params:     dict of query parameters e.g. {"user_id": 42, "status": "active"}
        chunksize:  if set, fetches in chunks and concatenates — useful for large results
    
    Example:
        _load_sql(conn, "SELECT * FROM users WHERE id = :user_id", params={"user_id": 42})
    """
    SUPPORTED_DIALECTS = [
        "postgresql", "mysql", "sqlite", "mssql",
        "mysql+pymysql", "mssql+pyodbc",
        "postgresql+psycopg2", "sqlite+pysqlite", "oracle+cx_oracle"
    ]

    database_index = connection.lower().find("://")
    if database_index == -1:
        raise ValueError("Invalid connection string: missing '://'")
    elif database_index < 5:
        raise ValueError("Invalid connection string: database type too short")

    database_type = connection[:database_index].lower()
    if database_type not in SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported database type: '{database_type}'. "
            f"Supported types: {', '.join(SUPPORTED_DIALECTS)}"
        )

    engine = create_engine(connection)

    # wrap query in SQLAlchemy's text() to enable safe parameter binding
    # this prevents SQL injection by never interpolating params as raw strings
    safe_query = text(query)

    try:
        with engine.connect() as conn:
            if chunksize:
                logger.info(f"Loading in chunks of {chunksize} rows")
                chunks = pd.read_sql(
                    safe_query,
                    con=conn,
                    params=params or {},
                    chunksize=chunksize
                )
                df = pd.concat(
                    (chunk for chunk in chunks),
                    ignore_index=True
                )
                logger.info(f"Chunks concatenated: {len(df)} total rows")
            else:
                df = pd.read_sql(
                    safe_query,
                    con=conn,
                    params=params or {}
                )

            logger.info(f"SQL query returned {len(df)} rows, {len(df.columns)} columns")
            return df

    except Exception as e:
        logger.warning(f"SQL load failed: {e}")
        raise


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

def _paginate_offset(url, params=None, headers=None, max_pages=50):
    """
    Keeps fetching pages until the API returns an empty list.
    """
    params = params or {}
    all_records = []
    page = 1

    while True:
        params["page"] = page
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        records = data if isinstance(data, list) else _find_records(data)
        # TODO: need to include maximum page limit or timeout to prevent infinite loops in case of API issues

        if not records:
            logger.info(f"Pagination complete. Total records fetched: {len(all_records)}")
            break

        logger.info(f"Fetched page {page} — {len(records)} records")
        all_records.extend(records)

        if page >= max_pages:
            logger.warning(f"Reached maximum page limit of {max_pages}. Stopping pagination.")
            break
        
        page += 1

    return all_records

def _paginate_cursor(url, params=None, headers=None, cursor_key="next_cursor"):
    """
    Follows cursor tokens until the API signals there are no more pages.
    """
    params = params or {}
    all_records = []

    while True:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        records = data if isinstance(data, list) else _find_records(data)
        if records:
            all_records.extend(records)

        next_cursor = data.get(cursor_key)

        if not next_cursor:
            logger.info(f"Cursor pagination complete. Total records: {len(all_records)}")
            break

        logger.info(f"Following cursor: {next_cursor}")
        params["cursor"] = next_cursor

    return all_records

def _paginate_link_header(url, params=None, headers=None, max_pages=50):
    """
    Follows 'Link' headers until there is no 'next' relation.
    GitHub's API is a good real-world example of this pattern.
    """
    all_records = []

    while url:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        records = data if isinstance(data, list) else _find_records(data)
        if records:
            all_records.extend(records)

        # TODO: research the 'Link' header format — it looks like:
        # <https://api.example.com/data?page=2>; rel="next"
        # how would you parse the next URL out of that string?
        # hint: look into the 'requests' library's link parsing,
        # or the 'parse_header_links' utility
        next_url = None  # replace this with real next URL extraction
        for link in response.links.values():
            if link.get("rel") == "next":
                next_url = link.get("url")
                break
        
        if not next_url:
            logger.info(f"Link pagination complete. Total records: {len(all_records)}")
            break

        if page >= max_pages:
            logger.warning(f"Hit max_pages limit of {max_pages}. There may be more records.")
            break

        logger.info(f"Following link header to page {page + 1}")

        # after the first request params are already baked into the next URL
        params = None
        url = next_url
        page += 1
        
    return all_records

def _load_api(url, params=None, headers=None, pagination=None, cursor_key="next_cursor", max_pages=50, **kwargs):
    """
    pagination options: None, "offset", "cursor", "link"
    """
 
    if pagination == "offset":
        records = _paginate_offset(url, params, headers, max_pages=max_pages) 
    elif pagination == "cursor":
        records = _paginate_cursor(url, params, headers, cursor_key=cursor_key)
    elif pagination == "link":
        records = _paginate_link_header(url, params, headers)
    elif pagination is None:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        records = data if isinstance(data, list) else _find_records(data)
        if records is None:
            logger.warning("Returning empty DataFrame due to unresolved response structure.")
            return pd.DataFrame()
    else:
        raise ValueError(f"Unsupported pagination type: {pagination}")

    return pd.DataFrame(records)


def _is_sql_connection(source):
    """
    Return True if source looks like a SQL connection string or SQLAlchemy engine.
    hint: what types or string patterns would indicate a DB connection?
    """
    raise NotImplementedError