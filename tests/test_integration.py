"""Opt-in integration tests that call the live Gemini API.

Run with:
    RUN_INTEGRATION=1 python -m pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from summarizer import PDFSummarizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to call the live Gemini API",
    ),
]

SAMPLE = (
    "The Orion program is a document-understanding research initiative that began "
    "in 2021. Its first milestone extracted text from scanned reports and raised "
    "accuracy from 71 percent to 94 percent. The second milestone added automatic "
    "summarization, cutting review time roughly in half. Future work targets "
    "multilingual documents and on-device processing."
)


def test_live_summary_returns_text():
    summarizer = PDFSummarizer()
    summary = summarizer.summarize_text(SAMPLE, "brief")
    assert isinstance(summary, str)
    assert len(summary) > 20


def test_live_pipeline_produces_combined_summary():
    summarizer = PDFSummarizer()
    chunks = [SAMPLE[:250], SAMPLE[250:]]
    result = summarizer.summarize_chunks(chunks, "bullet_points")
    assert result["total_chunks"] == 2
    assert any(item["ok"] for item in result["individual_summaries"])
    assert result["combined_summary"]


def test_live_quotes_are_extracted():
    summarizer = PDFSummarizer()
    quotes = summarizer.extract_key_quotes(SAMPLE)
    assert isinstance(quotes, list)
