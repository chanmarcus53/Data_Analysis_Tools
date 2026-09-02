from toolbox.ingestion.report import report, _print_console, _export_html, _export_excel
import toolbox.ingestion.report as report_module
from toolbox.ingestion import profiler as pf

import pytest
import pandas as pd

class TestPrintConsole:
    def test_prints_shape(self, capsys, profile_result):
        _print_console(profile_result)
        
        # capsys.readouterr() captures everything printed so far
        captured = capsys.readouterr()
        
        # captured.out is the printed string — assert it contains expected content
        assert "3" in captured.out        # row count
        assert "4" in captured.out        # column count
        assert "0.002" in captured.out    # memory

    def test_prints_warnings(self, capsys, profile_result):
        _print_console(profile_result)
        captured = capsys.readouterr()
        assert "high null percentage" in captured.out

    def test_prints_numeric_column(self, capsys, profile_result):
        _print_console(profile_result)
        captured = capsys.readouterr()
        assert "id" in captured.out
        assert "2.0" in captured.out      # mean

    def test_prints_categorical_column(self, capsys, profile_result):
        _print_console(profile_result)
        captured = capsys.readouterr()
        assert "status" in captured.out
        assert "active" in captured.out   # top value


class TestExportExcel:
    def test_exports_excel(self, profile_result, tmp_path):
        path = str(tmp_path / "report.xlsx")
        _export_excel(profile_result, path=path)
        assert (tmp_path / "report.xlsx").exists()

    def test_excel_has_correct_sheets(self, profile_result, tmp_path):
        path = str(tmp_path / "report.xlsx")
        _export_excel(profile_result, path=path)

        # read back and check sheets exist
        xl = pd.ExcelFile(path)
        assert "Overview" in xl.sheet_names
        assert "Warnings" in xl.sheet_names
        assert "Numeric Columns" in xl.sheet_names
        assert "Categorical Columns" in xl.sheet_names

    def test_excel_overview_has_correct_values(self, profile_result, tmp_path):
        path = str(tmp_path / "report.xlsx")
        _export_excel(profile_result, path=path)

        overview_df = pd.read_excel(path, sheet_name="Overview")
        assert overview_df["rows"].iloc[0] == 3
        assert overview_df["memory_mb"].iloc[0] == 0.002


class TestExportExcel:
    def test_exports_excel(self, profile_result, tmp_path):
        path = str(tmp_path / "report.xlsx")
        _export_excel(profile_result, path=path)
        assert (tmp_path / "report.xlsx").exists()

    def test_excel_has_correct_sheets(self, profile_result, tmp_path):
        path = str(tmp_path / "report.xlsx")
        _export_excel(profile_result, path=path)

        # read back and check sheets exist
        xl = pd.ExcelFile(path)
        assert "Overview" in xl.sheet_names
        assert "Warnings" in xl.sheet_names
        assert "Numeric Columns" in xl.sheet_names
        assert "Categorical Columns" in xl.sheet_names

    def test_excel_overview_has_correct_values(self, profile_result, tmp_path):
        path = str(tmp_path / "report.xlsx")
        _export_excel(profile_result, path=path)

        overview_df = pd.read_excel(path, sheet_name="Overview")
        assert overview_df["rows"].iloc[0] == 3
        assert overview_df["memory_mb"].iloc[0] == 0.002


class TestReport:
    def test_report_default_calls_console(self, profile_result, monkeypatch):
        called = []
        monkeypatch.setattr(report_module, "_print_console", lambda r: called.append("console"))
        report(profile_result)
        assert "console" in called

    def test_report_html_calls_export_html(self, profile_result, monkeypatch):
        called = []
        monkeypatch.setattr(report_module, "_export_html", lambda r, path=None: called.append("html"))
        report(profile_result, output="html")
        assert "html" in called

    def test_report_excel_calls_export_excel(self, profile_result, monkeypatch):
        called = []
        monkeypatch.setattr(report_module, "_export_excel", lambda r, path=None: called.append("excel"))
        report(profile_result, output="excel")
        assert "excel" in called

    def test_unsupported_format_raises(self, profile_result):
        with pytest.raises(ValueError):
            report(profile_result, output="pdf")