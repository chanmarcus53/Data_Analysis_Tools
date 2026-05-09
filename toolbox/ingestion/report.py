import profiler as pf
from toolbox.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def report(profile_result, output=None):
    """
    Sends a report about the data ingestion to the user

    Parameters:
    profile_result: dict from profiler.py

    Return:
    output=None prints to console.
    output="html" or output="excel" saves a file.
    """
    if output is None:
        _print_console(profile_result)
    elif output == "html":
        _export_html(profile_result)
    elif output == "excel":
        _export_excel(profile_result)
    else:
        raise ValueError(f"Unsupported output format: {output}")


def _print_console(profile_result):
    # TODO: print a readable summary — think about what a person
    # actually needs to see first when they load a new dataset
    print("----------- Ingestion Summary -------------------")
    print(f" Memory: {profile_result["memory"]}MB Ingested ")
    print(f" Data Shape: {profile_result["shape"][0]} X {profile_result['shape'][1]}")

    if profile_result["warnings"]:
        print("⚠ Warnings:")
        for warning in profile_result["warnings"]:
            print(f"  - {warning}")
    else:
        print("✓ No issues detected")
    
    print("\n----------- Column Profiles ---------------------")
    for key, value in profile_result['columns'].items():
        print()
        if "mean" in value:
            print(f"[Numeric] {key}")
            print(f"  Mean: {value['mean']}  Std: {value['std']}  Skewness: {value['skewness']}")
            print(f"  Min: {value['min']}  Max: {value['max']}")
            print(f"  Nulls: {value['null_count']} ({value['null_pct']}%)")
        else:
            print(f"[Categorical] {key}")
            print(f"  Unique values: {value['unique_count']}")
            print(f"  Nulls: {value['null_count']} ({value['null_pct']}%)")
            print(f"  Top values:")
            for col_value, count in value["top_values"].items():
                print(f"    {col_value}: {count}")
        print("------------------------------------------------------")
    
    print()

def _export_html(profile_result, path="report.html"):
    rows = ""
    for col_name, value in profile_result["columns"].items():
        if "mean" in value:
            rows += f"""
            <tr>
                <td>{col_name}</td>
                <td>Numeric</td>
                <td>{value['null_pct']}%</td>
                <td>{value['mean']}</td>
                <td>{value['std']}</td>
                <td>{value['min']} - {value['max']}</td>
                <td>{value['skewness']}</td>
            </tr>
            """
        else:
            rows += f"""
            <tr>
                <td>{col_name}</td>
                <td>Categorical</td>
                <td>{value['null_pct']}%</td>
                <td colspan="4">{value['unique_count']} unique values</td>
            </tr>
            """
        warnings_html = f""
        if profile_result["warnings"]:
            warning_items = "".join(f"<li>{w}</li>" for w in profile_result["warnings"])
            warnings_html = f"<ul>{warning_items}</ul>"
        else:
            warnings_html = f"<p>✓ No issues detected</p>"
        html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Ingestion Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th {{ background-color: #2c3e50; color: white; padding: 8px; text-align: left; }}
                    td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    li {{ color: #e74c3c; }}
                    p {{ color: #27ae60; }}
                </style>
            </head>
            <body>
                <h1>Ingestion Report</h1>

                <h2>Summary</h2>
                <p>Rows: {profile_result['shape'][0]} | Columns: {profile_result['shape'][1]} | Memory: {profile_result['memory']}MB</p>

                <h2>Warnings</h2>
                {warnings_html}

                <h2>Column Profiles</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Null %</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min - Max</th>
                        <th>Skewness</th>
                    </tr>
                    {rows}
                </table>
            </body>
            </html>
            """
        with open(path, "w") as f:
            f.write(html)

        logger.info(f"HTML report saved to {path}")

def _export_excel(profile_result, path="report.xlsx"):
    col = profile_result["columns"]

    overview_df = pd.DataFrame([{
        "rows": profile_result["shape"][0],
        "columns": profile_result["shape"][1],
        "memory_mb": profile_result["memory"]
    }])

    warnings_df = pd.DataFrame(
        profile_result["warnings"] if profile_result["warnings"] else ["No issues detected"],
        columns="warnings"
    )

    numeric = {k: v for k, v in col.items() if "mean" in v}
    categorical = {k: v for k, v in col.items() if "mean" not in v}

    numeric_df = pd.DataFrame.from_dict(numeric, orient="index") if numeric else pd.DataFrame()
    categorical_df = pd.DataFrame.from_dict(categorical, orient="index") if categorical else pd.DataFrame()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Overview", index=False)
        warnings_df.to_excel(writer, sheet_name="Warnings", index=False)
        numeric_df.to_excel(writer, sheet_name="Numeric Columns")
        categorical_df.to_excel(writer, sheet_name="Categorical Columns")

    logger.info(f"Excel report saved to {path}")