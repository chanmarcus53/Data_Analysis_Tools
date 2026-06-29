from toolbox.cleaning.audit import AuditTrail
import pandas as pd
import pytest

class TestLog:
    def test_log_adds_entry(self, audit_trail):
        audit_trail.log("handle_missing", "age", "Imputed 3 nulls with median value 30.0")
        assert len(audit_trail) == 1

    def test_log_captures_correct_fields(self, audit_trail):
        audit_trail.log("handle_missing", "age", "Imputed 3 nulls with median value 30.0")
        record = audit_trail.trail[0]
        assert record["step"] == "handle_missing"
        assert record["column"] == "age"
        assert record["details"] == "imputed 3 nulls with median value 30.0"

    def test_log_captures_timestamp(self, audit_trail):
        audit_trail.log("handle_missing", "age", "test")
        record = audit_trail.trail[0]
        assert "timestamp" in record
        assert isinstance(record["timestamp"], pd.Timestamp)

    def test_log_multiple_entries(self, audit_trail):
        audit_trail.log("handle_missing", "age", "first_step")
        audit_trail.log("handle_missing", "age", "second_step")
        assert len(audit_trail) == 2


class TestSummary:
    def test_emtpy_audit_trail(self, audit_trail):
        audit_trail.summary()
        assert len(audit_trail) == 0
    
    def test_summary_prints_entries(self, audit_trail, capsys):
        audit_trail.log("handle_missing", "age", "Imputed 3 nulls with median 30.0")
        audit_trail.summary()
        captured = capsys.readouterr()
        assert "handle_missing" in captured.out
        assert "age" in captured.out
    
    def test_summary_multiple_entries(self, audit_trail, capsys):
        audit_trail.log("handle_missing", "age", "first_step")
        audit_trail.log("handle_outliers", "age", "second_step")
        audit_trail.summary()
        captured = capsys.readouterr()
        assert "handle_missing" in captured.out
        assert "handle_outliers" in captured.out
        assert "2" in captured.out

class TestExport:
    def test_export_excel(self, audit_trail, tmp_path):
        audit_trail.log("handle_missing", "age", "first_step")
        path = str(tmp_path / "audit.xlsx")
        audit_trail.export(path=path)
        assert (tmp_path / "audit.xlsx").exists()

    def test_export_excel_correct_columns(self, audit_trail, tmp_path):
        audit_trail.log("handle_missing", "age", "test_details")
        path = str(tmp_path / "audit.xlsx")
        audit_trail.export(output="excel", path=path)
        df = pd.read_excel(path)
        assert "step" in df.columns
        assert "column" in df.columns
        assert "details" in df.columns
        assert "timestamp" in df.columns

    def test_export_excel_correct_values(self, audit_trail, tmp_path):
        audit_trail.log("handle_missing", "age", "test_details")
        path = str(tmp_path / "audit.html")
        audit_trail.export(output="html", path=path)
        df = pd.read_excel(path)
        assert df["step"].iloc[0] == "handle_missing"
        assert df["column"].iloc[0] == "age"

    def test_export_html_file_exists(self, audit_trail, tmp_path):
        audit_trail.log("handle_missing", "age", "test_details")
        path = str(tmp_path / "audit_html")
        audit_trail.export(output="html", path=path)
        assert (tmp_path / "audit.html").exists()

    def test_export_html_contains_data(self, audit_trail, tmp_path):
        audit_trail.log("handle_missing", "age", "test details")
        path = str(tmp_path / "audit.html")
        audit_trail.export(output="html", path=path)
        with open(path, "r") as f:
            content = f.read()
        assert "handle_missing" in content
        assert "age" in content

    def test_unsupported_format_raises(self, audit_trail):
        with pytest.raises(ValueError):
            audit_trail.export(output="pdf")

class TestClear:
    def test_clear_empties_trail(self, audit_trail):
        raise NotImplementedError

    def test_clear_on_empty_trail(self, audit_trail):
        raise NotImplementedError

class TestHelpers:
    def test_len_empty(self, audit_trail):
        raise NotImplementedError

    def test_len_after_logging(self, audit_trail):
        raise NotImplementedError

    def test_to_dataframe_returns_dataframe(self, audit_trail):
        raise NotImplementedError

    def test_to_dataframe_has_correct_columns(self, audit_trail):
        raise NotImplementedError