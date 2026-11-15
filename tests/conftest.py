"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from fpdf import FPDF

from pdf_processor import PDFProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SAMPLE_TEXT = (
    "The Orion program is a long-term research initiative focused on document "
    "understanding. It began in 2021 with a small team of engineers.\n\n"
    "The first milestone was a pipeline that extracted text from scanned reports. "
    "Accuracy improved from 71 percent to 94 percent over eighteen months.\n\n"
    "The second milestone introduced automatic summarization. Reviewers reported "
    "that reading time dropped by roughly half while comprehension stayed stable.\n\n"
    "Future work will focus on multilingual documents and on-device processing. "
    "The team recommends expanding the pilot to two additional departments."
)


def build_pdf_bytes(
    text: str = SAMPLE_TEXT,
    title: str = "Orion Program Report",
    author: str = "Test Author",
    subject: str = "Automated test fixture",
) -> bytes:
    """Create a small, text-based PDF entirely in memory."""
    pdf = FPDF()
    pdf.set_title(title)
    pdf.set_author(author)
    pdf.set_subject(subject)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for paragraph in text.split("\n\n"):
        pdf.multi_cell(0, 8, paragraph)
        pdf.ln(3)
    return bytes(pdf.output())


@pytest.fixture(scope="session")
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture(scope="session")
def pdf_bytes() -> bytes:
    return build_pdf_bytes()


@pytest.fixture()
def processor() -> PDFProcessor:
    return PDFProcessor()


@pytest.fixture()
def summary_payload() -> dict:
    return {
        "summary_type": "detailed",
        "total_chunks": 2,
        "combined_summary": "Combined summary text.",
        "individual_summaries": [
            {
                "chunk_number": 1,
                "summary": "First chunk summary.",
                "ok": True,
                "original_length": 1000,
                "summary_length": 250,
            },
            {
                "chunk_number": 2,
                "summary": "Second chunk failed.",
                "ok": False,
                "original_length": 800,
                "summary_length": 0,
            },
        ],
    }
