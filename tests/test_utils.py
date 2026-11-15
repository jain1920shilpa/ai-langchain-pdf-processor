"""Tests for export and estimation helpers (UI rendering is not exercised)."""

from __future__ import annotations

import pandas as pd

from utils import (
    create_summary_dataframe,
    estimate_processing_time,
    export_summary_to_text,
    summary_stem,
)


class TestExport:
    def test_text_export_contains_header_and_metadata(self, summary_payload):
        report = export_summary_to_text(summary_payload, "report.pdf")
        assert "PDF Summary Report" in report
        assert "File: report.pdf" in report
        assert "Summary Type: Detailed" in report
        assert "Total Chunks Processed: 2" in report

    def test_text_export_includes_both_chunks(self, summary_payload):
        report = export_summary_to_text(summary_payload)
        assert "Chunk 1:" in report
        assert "First chunk summary." in report
        assert "Chunk 2 (FAILED):" in report

    def test_text_export_without_filename(self, summary_payload):
        assert "File:" not in export_summary_to_text(summary_payload)


class TestDataframe:
    def test_shape_and_columns(self, summary_payload):
        frame = create_summary_dataframe(summary_payload)
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 2
        assert list(frame.columns) == [
            "Chunk",
            "Original Length",
            "Summary Length",
            "Compression Ratio",
            "Status",
        ]

    def test_compression_ratio_and_status(self, summary_payload):
        frame = create_summary_dataframe(summary_payload)
        assert frame.loc[0, "Compression Ratio"] == 0.25
        assert frame.loc[1, "Compression Ratio"] == 0
        assert frame.loc[0, "Status"] == "Success"
        assert frame.loc[1, "Status"] == "Error"

    def test_zero_original_length_does_not_divide_by_zero(self):
        payload = {
            "summary_type": "brief",
            "total_chunks": 1,
            "combined_summary": "x",
            "individual_summaries": [
                {
                    "chunk_number": 1,
                    "summary": "y",
                    "ok": True,
                    "original_length": 0,
                    "summary_length": 10,
                }
            ],
        }
        assert create_summary_dataframe(payload).loc[0, "Compression Ratio"] == 0


class TestEstimates:
    def test_seconds_format(self):
        assert estimate_processing_time(4, concurrency=4) == "~8 seconds"

    def test_minutes_format(self):
        assert estimate_processing_time(100, concurrency=4) == "~1 min 20s"

    def test_minimum_is_five_seconds(self):
        assert estimate_processing_time(0) == "~5 seconds"

    def test_zero_concurrency_is_treated_as_one(self):
        assert estimate_processing_time(1, concurrency=0) == "~8 seconds"


class TestFilename:
    def test_strips_pdf_suffix(self):
        assert summary_stem("annual_report.pdf") == "annual_report"

    def test_handles_name_without_suffix(self):
        assert summary_stem("report") == "report"

    def test_handles_dots_in_name(self):
        assert summary_stem("q3.2026.final.pdf") == "q3.2026.final"
