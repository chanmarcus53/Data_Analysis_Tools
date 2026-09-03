from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from toolbox.ingestion.loaders import load, _find_records, _is_sql_connection
from requests.exceptions import HTTPError
import json

class TestFindRecords:
    def test_finds_results_key(self, sample_api_response):
        result = _find_records(sample_api_response)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_returns_none_on_no_match(self):
        result = _find_records({"total": 10, "payload": []})
        assert result is None

    def test_wraps_single_dict_in_list(self):
        result = _find_records({"results": {"id": 1, "name": "Alice"}})
        assert isinstance(result, list)

class TestLoadFile:
    def test_load_csv(self, tmp_path):
        # tmp_path is a built in pytest fixture that creates a temporary directory
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob")

        df = load(str(csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
        assert df["id"].iloc[0] == 1
        assert df["name"].iloc[1] == "Bob"

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("some content")
        with pytest.raises(ValueError):
            load(str(bad_file))

    def test_load_excel(self, tmp_path):
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        df.to_excel(excel_file, index=False)
        loaded_df = load(str(excel_file))
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == ["id", "name"]
        assert loaded_df["id"].iloc[1] == 2
        assert loaded_df["name"].iloc[0] == "Alice"

    def test_load_json(self, tmp_path):
        json_file = tmp_path / "test.json"
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        json_file.write_text(json.dumps(data))
        loaded_df = load(str(json_file))
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == ["id", "name"]
        assert loaded_df["id"].iloc[1] == 2
        assert loaded_df["name"].iloc[0] == "Alice"

class TestLoadApi:
    @patch("toolbox.ingestion.loaders.requests.get")
    def test_list_response(self, mock_get):
        # set up the mock to return a fake response
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": 1}, {"id": 2}]
        mock_get.return_value = mock_response

        df = load("http://fake-api.com/data")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    @patch("toolbox.ingestion.loaders.requests.get")
    def test_dict_response(self, mock_get, sample_api_response):
        mock_response = MagicMock()
        mock_response.json.return_value = sample_api_response
        mock_get.return_value = mock_response

        df = load("http://fake-api.com/data")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
        assert df["id"].iloc[0] == 1
        assert df["name"].iloc[1] == "Bob"

    @patch("toolbox.ingestion.loaders.requests.get")
    def test_failed_request_raises(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            load("http://fake-api.com/data")


class TestIsSqlConnection:
    def test_valid_postgresql(self):
        assert _is_sql_connection("postgresql://user:pass@localhost/db") == True

    def test_valid_sqlite(self):
        assert _is_sql_connection("sqlite:///mydb.sqlite") == True

    def test_http_is_not_sql(self):
        assert _is_sql_connection("http://example.com/api") == False

    def test_file_path_is_not_sql(self):
        assert _is_sql_connection("/path/to/file.csv") == False