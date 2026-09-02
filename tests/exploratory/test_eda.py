from toolbox.exploratory.eda import run_eda, _collect_results, _print_console, _export_html, _export_excel
import pandas as pd
import pytest

class TestCollectResults:
    def test_returns_dict(self, sample_df):
        result = _collect_results(sample_df)
        assert isinstance(result, dict)

    def test_has_all_keys(self, sample_df):
        result = _collect_results(sample_df)
        assert "profile" in result
        assert "stats" in result
        assert "comparisons" in result

    def test_comparison_none_by_default(self, sample_df):
        result = _collect_results(sample_df)
        assert result["comparisons"] is None

    def test_comparison_with_group(self, categorical_df):
        result = _collect_results(categorical_df, compare_by={"column": "grade", "by": "school"})
        assert result["comparisons"] is not None
        assert "groups" in result["comparisons"]

    def test_comparison_with_periods(self, time_series_df):
        result = _collect_results(time_series_df, compare_periods={
            "column": "grade", "period_col": "quarter",
            "period_a": "Q1", "period_b": "Q2"
        })
        assert result["comparisons"] is not None
        assert "periods" in result["comparisons"]

class TestPrintConsole:
    def test_prints_overview(self, eda_result, capsys):
        _print_console(eda_result)
        captured = capsys.readouterr()
        assert "Overview" in captured.out

    def test_prints_shape(self, eda_result, capsys):
        _print_console(eda_result)
        captured = capsys.readouterr()
        assert "rows" in captured.out
        assert "columns" in captured.out

    def test_prints_skewness(self, eda_result, capsys):
        _print_console(eda_result)
        captured = capsys.readouterr()
        assert "Skewness" in captured.out

    def test_no_comparison_section_when_none(self, eda_result, capsys):
        _print_console(eda_result)
        captured = capsys.readouterr()
        assert "Comparison" not in captured.out

class TestExportHtml:
    def test_file_exists(self, eda_result, tmp_path):
        path = str(tmp_path / "eda.html")
        _export_html(eda_result, path)
        assert (tmp_path / "eda.html").exists()

    def test_contains_overview(self, eda_result, tmp_path):
        path = str(tmp_path / "eda.html")
        _export_html(eda_result, path)
        with open(path, "r") as f:
            content = f.read()
        assert "Overview" in content
        assert "Skewness" in content
        assert "Percentiles" in content

    def test_contains_correlations(self, eda_result, tmp_path):
        path = str(tmp_path / "eda.html")
        _export_html(eda_result, path)
        with open(path, "r") as f:
            content = f.read()
        assert "Pearson" in content
        assert "Spearman" in content
        assert "Kendall" in content

class TestExportExcel:
    def test_file_exists(self, eda_result, tmp_path):
        path = str(tmp_path / "eda.xlsx")
        _export_excel(eda_result, path)
        assert (tmp_path / "eda.xlsx").exists()

    def test_has_correct_sheets(self, eda_result, tmp_path):
        path = str(tmp_path / "eda.xlsx")
        _export_excel(eda_result, path)
        xl = pd.ExcelFile(path)
        assert "Overview" in xl.sheet_names
        assert "Skewness & Kurtosis" in xl.sheet_names
        assert "Percentiles" in xl.sheet_names
        assert "Value Counts" in xl.sheet_names
        assert "Pearson Correlation" in xl.sheet_names

class TestRunEda:
    def test_console_output(self, sample_df, capsys):
        run_eda(sample_df)
        captured = capsys.readouterr()
        assert "EDA" in captured.out

    def test_html_output(self, sample_df, tmp_path):
        path = str(tmp_path / "eda.html")
        run_eda(sample_df, output="html", path=path)
        assert (tmp_path / "eda.html").exists()

    def test_excel_output(self, sample_df, tmp_path):
        path = str(tmp_path / "eda.xlsx")
        run_eda(sample_df, output="excel", path=path)
        assert (tmp_path / "eda.xlsx").exists()

    def test_html_without_path_raises(self, sample_df):
        with pytest.raises(ValueError):
            run_eda(sample_df, output="html")

    def test_excel_without_path_raises(self, sample_df):
        with pytest.raises(ValueError):
            run_eda(sample_df, output="excel")

    def test_invalid_output_raises(self, sample_df):
        with pytest.raises(ValueError):
            run_eda(sample_df, output="pdf")

    def test_with_group_comparison(self, categorical_df, tmp_path):
        path = str(tmp_path / "eda.html")
        run_eda(categorical_df, output="html", path=path,
                compare_by={"column": "grade", "by": "school"})
        assert (tmp_path / "eda.html").exists()