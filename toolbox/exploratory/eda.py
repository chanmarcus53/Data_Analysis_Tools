from toolbox.logger import get_logger
from toolbox.ingestion.profiler import profile
from toolbox.exploratory.stats import summarise
from toolbox.exploratory.comparison import compare
import pandas as pd

logger = get_logger(__name__)

def run_eda(df, output=None, path=None, compare_by=None, compare_periods=None):
    # entry point — orchestrates full EDA pipeline
    # calls profiler, stats, and comparison
    # outputs to console, HTML, or Excel
    results = _collect_results(df, compare_by, compare_periods)
    
    if output is None:
        _print_console(results)
    elif output == "html":
        if path is None:
            raise ValueError("Path must be provided for HTML output.")
        _export_html(results, path)
    elif output == "excel":
        if path is None:
            raise ValueError("Path must be provided for Excel output.")
        _export_excel(results, path)
    else:
        raise ValueError(f"Invalid output type: '{output}'. Choose from 'html' or 'excel'.")


def _collect_results(df, compare_by=None, compare_periods=None):
    # gathers all results into one dict
    # profile_result, stats_result, comparison_result
    result = {
        "profile": profile(df),
        "stats": summarise(df),
        "comparisons": None
    }

    if compare_by is not None:
        result["comparisons"] = compare(df, **compare_by)
    elif compare_periods is not None:
        result["comparisons"] = compare(df, **compare_periods)
    return result

def _print_console(eda_result):
    print("=========== EDA Report ===========")
    
    # profile section
    profile = eda_result["profile"]
    print(f"\n--- Overview ---")
    print(f"Shape: {profile['shape'][0]} rows x {profile['shape'][1]} columns")
    print(f"Memory: {profile['memory']}MB")
    
    if profile["warnings"]:
        print("\n⚠ Warnings:")
        for w in profile["warnings"]:
            print(f"  - {w}")

    # stats section
    stats = eda_result["stats"]
    print(f"\n--- Skewness & Kurtosis ---")
    for col in stats["skewness"]:
        print(f"  {col}: skewness={stats['skewness'][col]}, kurtosis={stats['kurtosis'][col]}")

    print(f"\n--- Percentiles ---")
    for col, percentiles in stats["percentiles"].items():
        print(f"  {col}: {percentiles}")

    print(f"\n--- Value Counts ---")
    for col, counts in stats["value_counts"].items():
        print(f"  {col}: {counts}")

    # comparison section
    if eda_result["comparisons"] is not None:
        comp = eda_result["comparisons"]
        print(f"\n--- Comparison ---")
        print(f"  {comp['significance']['interpretation']}")

def _export_excel(eda_result, path):
    profile_result = eda_result["profile"]
    stats_result = eda_result["stats"]

    # overview sheet
    overview_df = pd.DataFrame([{
        "rows": profile_result["shape"][0],
        "columns": profile_result["shape"][1],
        "memory_mb": profile_result["memory"]
    }])

    # warnings sheet
    warnings_df = pd.DataFrame(
        profile_result["warnings"] if profile_result["warnings"] else ["No issues detected"],
        columns=["warnings"]
    )

    # skewness and kurtosis sheet
    skew_kurt_df = pd.DataFrame({
        "column": list(stats_result["skewness"].keys()),
        "skewness": list(stats_result["skewness"].values()),
        "kurtosis": list(stats_result["kurtosis"].values())
    })

    # percentiles sheet
    percentiles_df = pd.DataFrame.from_dict(
        stats_result["percentiles"],
        orient="index"
    )

    # value counts sheet — flatten into a readable table
    value_counts_rows = []
    for col, counts in stats_result["value_counts"].items():
        for value, count in counts.items():
            value_counts_rows.append({
                "column": col,
                "value": value,
                "count": count
            })
    value_counts_df = pd.DataFrame(value_counts_rows) if value_counts_rows else pd.DataFrame()

    # correlations — one sheet per method
    pearson_df = pd.DataFrame.from_dict(stats_result["correlations"]["pearson"])
    spearman_df = pd.DataFrame.from_dict(stats_result["correlations"]["spearman"])
    kendall_df = pd.DataFrame.from_dict(stats_result["correlations"]["kendall"])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Overview", index=False)
        warnings_df.to_excel(writer, sheet_name="Warnings", index=False)
        skew_kurt_df.to_excel(writer, sheet_name="Skewness & Kurtosis", index=False)
        percentiles_df.to_excel(writer, sheet_name="Percentiles")
        value_counts_df.to_excel(writer, sheet_name="Value Counts", index=False)
        pearson_df.to_excel(writer, sheet_name="Pearson Correlation")
        spearman_df.to_excel(writer, sheet_name="Spearman Correlation")
        kendall_df.to_excel(writer, sheet_name="Kendall Correlation")

        # optional comparison sheet
        if eda_result["comparisons"] is not None:
            comp = eda_result["comparisons"]
            comp_df = pd.DataFrame([{
                "column": comp["column"],
                "test_used": comp["significance"]["test_used"],
                "p_value": comp["significance"]["p_value"],
                "significant": comp["significance"]["significant"],
                "interpretation": comp["significance"]["interpretation"]
            }])
            comp_df.to_excel(writer, sheet_name="Comparison", index=False)

    logger.info(f"EDA Excel report saved to {path}")
    

def _export_html(eda_result, path):
    profile_result = eda_result["profile"]
    stats_result = eda_result["stats"]

    # warnings section
    if profile_result["warnings"]:
        warning_items = "".join(f"<li>{w}</li>" for w in profile_result["warnings"])
        warnings_html = f"<ul>{warning_items}</ul>"
    else:
        warnings_html = "<p>✓ No issues detected</p>"

    # skewness and kurtosis table
    skew_kurt_rows = ""
    for col in stats_result["skewness"]:
        skew_kurt_rows += f"""
        <tr>
            <td>{col}</td>
            <td>{stats_result['skewness'][col]}</td>
            <td>{stats_result['kurtosis'][col]}</td>
        </tr>
        """

    # percentiles table
    percentile_headers = "".join(f"<th>{k}</th>" for k in ["5%", "10%", "25%", "50%", "75%", "90%", "95%"])
    percentile_rows = ""
    for col, values in stats_result["percentiles"].items():
        cells = "".join(f"<td>{round(v, 4)}</td>" for v in values.values())
        percentile_rows += f"<tr><td>{col}</td>{cells}</tr>"

    # value counts table
    value_count_rows = ""
    for col, counts in stats_result["value_counts"].items():
        for value, count in counts.items():
            value_count_rows += f"""
            <tr>
                <td>{col}</td>
                <td>{value}</td>
                <td>{count}</td>
            </tr>
            """

    # correlation tables
    def _dict_to_html_table(d):
        cols = list(d.keys())
        headers = "<th></th>" + "".join(f"<th>{c}</th>" for c in cols)
        rows = ""
        for row_col in cols:
            cells = "".join(f"<td>{round(d[c].get(row_col, 0), 4)}</td>" for c in cols)
            rows += f"<tr><td>{row_col}</td>{cells}</tr>"
        return f"<table><tr>{headers}</tr>{rows}</table>"

    # comparison section
    comparison_html = ""
    if eda_result["comparisons"] is not None:
        comp = eda_result["comparisons"]
        comparison_html = f"""
        <h2>Comparison</h2>
        <p><strong>Test used:</strong> {comp['significance']['test_used']}</p>
        <p><strong>P-value:</strong> {comp['significance']['p_value']}</p>
        <p><strong>Significant:</strong> {comp['significance']['significant']}</p>
        <p><strong>Interpretation:</strong> {comp['significance']['interpretation']}</p>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EDA Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th {{ background-color: #2c3e50; color: white; padding: 8px; text-align: left; }}
            td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            li {{ color: #e74c3c; }}
            p {{ color: #27ae60; }}
        </style>
    </head>
    <body>
        <h1>EDA Report</h1>

        <h2>Overview</h2>
        <p>Rows: {profile_result['shape'][0]} | 
           Columns: {profile_result['shape'][1]} | 
           Memory: {profile_result['memory']}MB</p>

        <h2>Warnings</h2>
        {warnings_html}

        <h2>Skewness & Kurtosis</h2>
        <table>
            <tr><th>Column</th><th>Skewness</th><th>Kurtosis</th></tr>
            {skew_kurt_rows}
        </table>

        <h2>Percentiles</h2>
        <table>
            <tr><th>Column</th>{percentile_headers}</tr>
            {percentile_rows}
        </table>

        <h2>Value Counts</h2>
        <table>
            <tr><th>Column</th><th>Value</th><th>Count</th></tr>
            {value_count_rows}
        </table>

        <h2>Pearson Correlation</h2>
        {_dict_to_html_table(stats_result['correlations']['pearson'])}

        <h2>Spearman Correlation</h2>
        {_dict_to_html_table(stats_result['correlations']['spearman'])}

        <h2>Kendall Correlation</h2>
        {_dict_to_html_table(stats_result['correlations']['kendall'])}

        {comparison_html}
    </body>
    </html>
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"EDA HTML report saved to {path}")