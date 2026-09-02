from toolbox.logger import get_logger
from scipy import stats
import pandas as pd

logger = get_logger(__name__)

def compare(df, column, by=None, period_col=None, period_a=None, period_b=None):
    """
    Entry point — dispatches to group or time comparison.
    
    Group comparison example:
        compare(df, column="age", by="status")
        
    Time period comparison example:
        compare(df, column="age", period_col="quarter", 
                period_a="Q1", period_b="Q2")
    """
    if by is not None and period_col is not None:
        raise ValueError("Pass either 'by' for group comparison or 'period_col' for time comparison, not both.")
    
    if by is not None:
        return _compare_groups(df, column, by)
    elif all(p is not None for p in [period_col, period_a, period_b]):
        return _compare_periods(df, column, period_col, period_a, period_b)
    else:
        raise ValueError("Must provide either 'by' or all of 'period_col', 'period_a', 'period_b'.")


def _compare_groups(df, column, by):
    """
    Compare distributions between all groups in a categorical column.
    """
    unique_groups = df[by].unique()
    logger.info(f"Comparing '{column}' across {len(unique_groups)} groups in '{by}'")

    # build summary stats per group
    group_stats = {}
    for group in unique_groups:
        group_data = df[df[by] == group][column].dropna()
        group_stats[group] = {
            "count": int(group_data.count()),
            "mean": round(group_data.mean(), 4),
            "std": round(group_data.std(), 4),
            "median": round(group_data.median(), 4),
            "min": round(group_data.min(), 4),
            "max": round(group_data.max(), 4)
        }
        logger.debug(f"Group '{group}': n={group_data.count()}, mean={group_data.mean():.4f}")

    # extract raw series for each group and pass to significance test
    groups = [df[df[by] == val][column].dropna() for val in unique_groups]
    significance = _test_significance(*groups, group_names=list(unique_groups))

    return {
        "column": column,
        "by": by,
        "groups": group_stats,
        "significance": significance
    }


def _compare_periods(df, column, period_col, period_a, period_b):
    """
    Compare distributions between two time periods.
    """
    # TODO: validate that period_a and period_b exist in period_col
    # hint: what should happen if they don't?
    
    if period_a not in df[period_col].values:
        raise ValueError(f"Period '{period_a}' not found in column '{period_col}'")
    if period_b not in df[period_col].values:
        raise ValueError(f"Period '{period_b}' not found in column '{period_col}'")

    group_a = df[df[period_col] == period_a][column].dropna()
    group_b = df[df[period_col] == period_b][column].dropna()

    logger.info(f"Comparing '{column}' between periods '{period_a}' and '{period_b}'")

    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError(f"One of the periods '{period_a}' or '{period_b}' has no data for column '{column}'")

    period_stats = {
        period_a: {
            "count": int(group_a.count()),
            "mean": round(group_a.mean(), 4),
            "std": round(group_a.std(), 4),
            "median": round(group_a.median(), 4),
            "min": round(group_a.min(), 4),
            "max": round(group_a.max(), 4)
        },
        period_b: {
            "count": int(group_b.count()),
            "mean": round(group_b.mean(), 4),
            "std": round(group_b.std(), 4),
            "median": round(group_b.median(), 4),
            "min": round(group_b.min(), 4),
            "max": round(group_b.max(), 4)
        }
    }

    significance = _test_significance(group_a, group_b, group_names=[period_a, period_b])

    return {
        "column": column,
        "period_col": period_col,
        "periods": period_stats,
        "significance": significance
    }

def _test_significance(*groups, group_names=None):
    """
    Automatically selects and runs the right significance test.
    2 groups → t-test or Mann-Whitney
    3+ groups → ANOVA or Kruskal-Wallis
    """
    if group_names is None:
        group_names = [f"group_{i}" for i in range(len(groups))]

    all_normal = all(_check_normality(g) for g in groups)

    if len(groups) == 2:
        if all_normal:
            statistic, p_value = stats.ttest_ind(groups[0], groups[1])
            test_used = "t-test"
        else:
            statistic, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            test_used = "mann-whitney"

    elif len(groups) >= 3:
        if all_normal:
            statistic, p_value = stats.f_oneway(*groups)
            test_used = "anova"
        else:
            statistic, p_value = stats.kruskal(*groups)
            test_used = "kruskal-wallis"
    else:
        raise ValueError("At least 2 groups are required for significance testing")

    significant = p_value < 0.05

    # TODO: write a plain English interpretation for each test type
    # hint: mention the test used, number of groups, p_value, and whether significant
    interpretation = _interpret(test_used, group_names, p_value, significant)

    logger.info(f"Significance test: {test_used}, p={p_value:.4f}, significant={significant}")

    return {
        "test_used": test_used,
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "significant": significant,
        "interpretation": interpretation
    }


def _interpret(test_used, group_names, p_value, significant):
    """
    Returns a plain English interpretation of the significance test result.
    """
    if test_used in ["t-test", "mann-whitney"]:
        message = f"Comparing two groups: {group_names[0]} and {group_names[1]}.\n"
    else:
        message = f"Comparing {len(group_names)} groups: {', '.join(str(g) for g in group_names)}.\n"

    if significant:
        message += (f"The {test_used} indicates a statistically significant difference "
                    f"between the groups (p={p_value:.4f}). "
                    f"This suggests the difference is unlikely to be due to chance.")
    else:
        message += (f"The {test_used} found no statistically significant difference "
                    f"between the groups (p={p_value:.4f}). "
                    f"This suggests any observed difference may be due to chance.")
    return message


def _check_normality(series):
    """
    Shapiro-Wilk for n < 5000, normaltest for larger samples.
    Returns True if normal, False if not.
    """
    series = series.dropna()

    if len(series) < 3:
        logger.warning(f"Sample too small for normality test (n={len(series)}), assuming non-normal")
        return False

    if len(series) < 5000:
        _, p_value = stats.shapiro(series)
    else:
        _, p_value = stats.normaltest(series)

    is_normal = p_value > 0.05
    logger.debug(f"Normality test: p={p_value:.4f}, normal={is_normal}")
    return is_normal