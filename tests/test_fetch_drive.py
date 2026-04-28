"""Tests for fetch_drive module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import fetch_drive


class TestGetAccessToken:
    def test_returns_token_on_success(self):
        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {"access_token": "test-token-abc"}

        with patch.object(fetch_drive.msal, "ConfidentialClientApplication", return_value=mock_app):
            token = fetch_drive.get_access_token()

        assert token == "test-token-abc"

    def test_raises_on_failure(self):
        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            "error": "invalid_client",
            "error_description": "Bad credentials",
        }

        with patch.object(fetch_drive.msal, "ConfidentialClientApplication", return_value=mock_app):
            with pytest.raises(Exception, match="Bad credentials"):
                fetch_drive.get_access_token()

    def test_confidential_app_constructed(self):
        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {"access_token": "tok"}

        with patch.object(fetch_drive.msal, "ConfidentialClientApplication", return_value=mock_app) as mock_cls:
            fetch_drive.get_access_token()

        mock_cls.assert_called_once()


class TestSyncDriveFiles:
    def test_skips_on_token_error(self, capsys):
        with patch.object(fetch_drive, "get_access_token", side_effect=Exception("auth failed")):
            fetch_drive.sync_drive_files()

        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    def test_skips_non_pdf_xlsx_files(self, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [
                {"name": "readme.txt", "@microsoft.graph.downloadUrl": "http://example.com/readme.txt"},
            ]
        }

        with patch.object(fetch_drive, "get_access_token", return_value="tok"), \
             patch.object(fetch_drive.requests, "get", return_value=mock_response), \
             patch.object(fetch_drive, "INPUT_DIR", tmp_path):
            fetch_drive.sync_drive_files()

        assert list(tmp_path.iterdir()) == []

    def test_downloads_pdf_files(self, tmp_path):
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.json.return_value = {
            "value": [
                {"name": "report.pdf", "@microsoft.graph.downloadUrl": "http://example.com/report.pdf"},
            ]
        }
        file_response = MagicMock()
        file_response.content = b"%PDF-test-content"

        call_count = {"n": 0}

        def _get(url, **kwargs):
            call_count["n"] += 1
            if "graph.microsoft.com" in url:
                return list_response
            return file_response

        with patch.object(fetch_drive, "get_access_token", return_value="tok"), \
             patch.object(fetch_drive.requests, "get", side_effect=_get), \
             patch.object(fetch_drive, "INPUT_DIR", tmp_path):
            fetch_drive.sync_drive_files()

        downloaded = list(tmp_path.iterdir())
        assert len(downloaded) == 1
        assert downloaded[0].name == "report.pdf"

    def test_skips_already_existing_files(self, tmp_path, capsys):
        existing = tmp_path / "existing.pdf"
        existing.write_bytes(b"old content")

        list_response = MagicMock()
        list_response.status_code = 200
        list_response.json.return_value = {
            "value": [
                {"name": "existing.pdf", "@microsoft.graph.downloadUrl": "http://example.com/existing.pdf"},
            ]
        }

        with patch.object(fetch_drive, "get_access_token", return_value="tok"), \
             patch.object(fetch_drive.requests, "get", return_value=list_response), \
             patch.object(fetch_drive, "INPUT_DIR", tmp_path):
            fetch_drive.sync_drive_files()

        captured = capsys.readouterr()
        assert "already exists" in captured.out or "Skipping" in captured.out
        assert existing.read_bytes() == b"old content"

    def test_handles_non_200_response(self, capsys):
        error_response = MagicMock()
        error_response.status_code = 403
        error_response.text = "Forbidden"

        with patch.object(fetch_drive, "get_access_token", return_value="tok"), \
             patch.object(fetch_drive.requests, "get", return_value=error_response):
            fetch_drive.sync_drive_files()

        captured = capsys.readouterr()
        assert "Error" in captured.out or "error" in captured.out.lower()

    def test_skips_item_without_download_url(self, tmp_path):
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.json.return_value = {
            "value": [
                {"name": "nodl.pdf"},  # no downloadUrl key
            ]
        }

        with patch.object(fetch_drive, "get_access_token", return_value="tok"), \
             patch.object(fetch_drive.requests, "get", return_value=list_response), \
             patch.object(fetch_drive, "INPUT_DIR", tmp_path):
            fetch_drive.sync_drive_files()

        assert list(tmp_path.iterdir()) == []
