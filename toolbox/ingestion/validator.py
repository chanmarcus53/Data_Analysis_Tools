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
    # TODO: check which expected columns are missing from df
    # hint: sets are useful here
    raise NotImplementedError

def _check_dtypes(df, expected_dtypes, results):
    # TODO: compare df[col].dtype against expected for each column
    # hint: pandas stores dtypes as strings — look into dtype.name
    raise NotImplementedError

def _check_nulls(df, non_nullable, results):
    # TODO: flag any column in non_nullable that contains nulls
    raise NotImplementedError

def _check_value_sets(df, value_sets, results):
    # TODO: for each column, check if any values fall outside the allowed set
    raise NotImplementedError