def compare(df, column, by=None, periods=None):
    # entry point — dispatches to group or time comparison
    pass

def _compare_groups(df, column, by):
    # compare distributions between groups
    # e.g. age distribution for active vs inactive
    pass

def _compare_periods(df, column, period_col, period_a, period_b):
    # compare distributions between two time periods
    pass

def _test_significance(group_a, group_b):
    # runs both parametric and non-parametric tests
    # t-test for normally distributed data
    # mann-whitney for non-normal data
    # returns test statistic, p-value, and interpretation
    pass

def _check_normality(series):
    # shapiro-wilk test to determine which significance test to use
    # hint: look into scipy.stats.shapiro
    pass