def run_eda(df, output=None, path=None):
    # entry point — orchestrates full EDA pipeline
    # calls profiler, stats, and comparison
    # outputs to console, HTML, or Excel
    pass

def _collect_results(df):
    # gathers all results into one dict
    # profile_result, stats_result, comparison_result
    pass

def _export_html(eda_result, path):
    # full HTML report with all sections
    pass

def _export_excel(eda_result, path):
    # one sheet per section
    # Overview, Stats, Correlations, Comparisons
    pass