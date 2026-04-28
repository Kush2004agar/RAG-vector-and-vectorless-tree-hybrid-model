"""Tests for run_qa_pipeline module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import run_qa_pipeline


class TestRunPipeline:
    def _make_excel(self, tmp_path: Path, data: dict) -> Path:
        df = pd.DataFrame(data)
        path = tmp_path / "questions.xlsx"
        df.to_excel(path, index=False)
        return path

    def test_no_excel_files_returns_early(self, tmp_path, capsys):
        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path):
            run_qa_pipeline.run_pipeline()

        captured = capsys.readouterr()
        assert "No Excel files" in captured.out

    def test_processes_question_column(self, tmp_path):
        self._make_excel(tmp_path, {"Question": ["What is SSH?", "What is TLS?", "What is HTTPS?"]})

        mock_answer = MagicMock(return_value="A mocked answer.")
        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path), \
             patch.object(run_qa_pipeline, "answer_question", mock_answer):
            run_qa_pipeline.run_pipeline()

        assert mock_answer.call_count == 3

    def test_falls_back_to_first_column_if_no_question_column(self, tmp_path, capsys):
        self._make_excel(tmp_path, {"Query": ["Q1", "Q2"]})

        mock_answer = MagicMock(return_value="answer")
        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path), \
             patch.object(run_qa_pipeline, "answer_question", mock_answer):
            run_qa_pipeline.run_pipeline()

        captured = capsys.readouterr()
        assert "Falling back" in captured.out or mock_answer.call_count >= 1

    def test_skips_empty_questions(self, tmp_path):
        self._make_excel(tmp_path, {"Question": ["", "  ", "Real question?"]})

        mock_answer = MagicMock(return_value="answer")
        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path), \
             patch.object(run_qa_pipeline, "answer_question", mock_answer):
            run_qa_pipeline.run_pipeline()

        # Empty and whitespace questions should be skipped (N/A returned)
        assert mock_answer.call_count <= 1

    def test_output_file_created(self, tmp_path):
        self._make_excel(tmp_path, {"Question": ["Q1"]})

        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path), \
             patch.object(run_qa_pipeline, "answer_question", return_value="ans"):
            run_qa_pipeline.run_pipeline()

        output_files = list(tmp_path.glob("answered_*.xlsx"))
        assert len(output_files) == 1

    def test_output_contains_generated_answer_column(self, tmp_path):
        self._make_excel(tmp_path, {"Question": ["Q1", "Q2"]})

        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path), \
             patch.object(run_qa_pipeline, "answer_question", return_value="my answer"):
            run_qa_pipeline.run_pipeline()

        output_files = list(tmp_path.glob("answered_*.xlsx"))
        df = pd.read_excel(output_files[0])
        assert "Generated_Answer" in df.columns

    def test_processes_at_most_5_questions(self, tmp_path):
        questions = [f"Question {i}?" for i in range(10)]
        self._make_excel(tmp_path, {"Question": questions})

        mock_answer = MagicMock(return_value="ans")
        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path), \
             patch.object(run_qa_pipeline, "answer_question", mock_answer):
            run_qa_pipeline.run_pipeline()

        assert mock_answer.call_count <= 5

    def test_bad_excel_file_handled_gracefully(self, tmp_path, capsys):
        bad_file = tmp_path / "bad.xlsx"
        bad_file.write_bytes(b"not an excel file at all")

        with patch.object(run_qa_pipeline, "INPUT_DIR", tmp_path):
            run_qa_pipeline.run_pipeline()

        captured = capsys.readouterr()
        assert "Failed" in captured.out or "Error" in captured.out or "error" in captured.out.lower()
