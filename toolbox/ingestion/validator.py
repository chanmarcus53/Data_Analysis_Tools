from toolbox.logger import get_logger

logger = get_logger(__name__)

def validate(df, schema=None):
    """
    Run all validation checks and return a report dict.
    Should never hard crash — capture failures as results.

    schema example:
    {
        "columns": ["id", "name", "date"],
        "dtypes": {"id": "int64", "date": "datetime64"},
        "non_nullable": ["id"],
        "value_sets": {"status": ["active", "inactive"]}
    }
    """
    results = {
        "passed": [],
        "warnings": [],
        "failed": []
    }

    if schema is None:
        return results

    _check_columns(df, schema.get("columns", []), results)
    _check_dtypes(df, schema.get("dtypes", {}), results)
    _check_nulls(df, schema.get("non_nullable", []), results)
    _check_value_sets(df, schema.get("value_sets", {}), results)

    return results


def _check_columns(df, expected_columns, results):
    """
    Check if all expected columns are present and if there are any extra columns.
    Parameters:
    - df: pandas DataFrame to validate
    - expected_columns: list of column names that are expected to be in the DataFrame
    - results: dict to store validation results (passed, warnings, failed)

    Returns:
    - list of missing columns or extra columns
    - messages for column issues are added to results dict
    """
    missing_columns = list(set(expected_columns) - set(df.columns))
    extra_columns = list(set(df.columns) - set(expected_columns))
    if missing_columns:
        msg = f"Missing columns: {missing_columns}"
        results["failed"].append(msg)
        logger.error(msg)
    if extra_columns:
        msg = f"Extra columns: {extra_columns}"
        results["warnings"].append(msg)
        logger.warning(msg)
    if not missing_columns and not extra_columns:
        results["passed"].append("All expected columns are present, no extras.")
        logger.info("All expected columns are present, no extras.")

def _check_dtypes(df, expected_dtypes, results):
    column_types = df.dtypes.apply(lambda x: x.name).to_dict()
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = column_types.get(col)
        if actual_dtype is None:
            msg = f"Column '{col}' is missing, cannot check dtype."
            results["failed"].append(msg)
            logger.error(msg)
        elif actual_dtype != expected_dtype:
            msg = f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'."
            results["failed"].append(msg)
            logger.error(msg)
        else:
            msg = f"Column '{col}' has correct dtype '{expected_dtype}'."
            results["passed"].append(msg)
            logger.info(msg)

def _check_nulls(df, non_nullable, results):
    for col in non_nullable:
        if col not in df.columns:
            msg = f"Column '{col}' is missing, cannot check for nulls."
            results["failed"].append(msg)
            logger.error(msg)
        elif df[col].isnull().any():
            msg = f"Column '{col}' contains null values but is marked as non-nullable."
            results["failed"].append(msg)
            logger.error(msg)
        else:
            msg = f"Column '{col}' contains no null values as expected."
            results["passed"].append(msg)
            logger.info(msg)

def _check_value_sets(df, value_sets, results):
    for col, allowed_values in value_sets.items():
        if col not in df.columns:
            msg = f"Column '{col}' is missing, cannot check value set."
            results["failed"].append(msg)
            logger.error(msg)
        elif not df[col].isin(allowed_values).all():
            invalid_values = df[~df[col].isin(allowed_values)][col].unique()
            msg = f"Column '{col}' contains invalid values: {invalid_values}. Allowed values are: {allowed_values}."
            results["failed"].append(msg)
            logger.error(msg)
        else:
            msg = f"All values in column '{col}' are within the allowed set."
            results["passed"].append(msg)
            logger.info(msg)
