"""Tests for the PDF extraction, cleaning and chunking pipeline."""

from __future__ import annotations

import io

import pytest

from pdf_processor import PDFProcessor


class TestCleaning:
    def test_removes_control_characters(self, processor):
        assert processor.clean_text("a\x00b\x07c\x1fd") == "abcd"

    def test_collapses_spaces_and_blank_lines(self, processor):
        raw = "First   line.  \n\n\n\n  Second    line."
        assert processor.clean_text(raw) == "First line.\n\nSecond line."

    def test_keeps_paragraph_breaks(self, processor):
        assert processor.clean_text("Para one.\n\nPara two.") == "Para one.\n\nPara two."

    def test_empty_text_stays_empty(self, processor):
        assert processor.clean_text("   \n\n  ") == ""


class TestTokens:
    def test_empty_text_is_zero_tokens(self, processor):
        assert processor.count_tokens("") == 0

    def test_counts_are_positive_and_monotonic(self, processor):
        short = processor.count_tokens("hello world")
        long = processor.count_tokens("hello world " * 50)
        assert 0 < short < long


class TestChunking:
    def test_empty_text_raises(self, processor):
        with pytest.raises(ValueError, match="No extractable text"):
            processor.chunk_text("   ")

    def test_rejects_non_positive_budget(self, processor):
        with pytest.raises(ValueError, match="max_tokens"):
            processor.chunk_text("Some text.", max_tokens=0)

    def test_chunks_respect_token_budget(self, processor, sample_text):
        chunks = processor.chunk_text(sample_text, max_tokens=40)
        assert len(chunks) > 1
        assert all(processor.count_tokens(chunk) <= 40 for chunk in chunks)

    def test_short_text_is_a_single_chunk(self, processor):
        chunks = processor.chunk_text("One short sentence.", max_tokens=100)
        assert chunks == ["One short sentence."]

    def test_oversized_sentence_is_hard_split(self, processor):
        long_sentence = " ".join(["word"] * 300) + "."
        chunks = processor.chunk_text(long_sentence, max_tokens=20)
        assert len(chunks) > 1
        assert all(processor.count_tokens(chunk) <= 20 for chunk in chunks)

    def test_no_text_is_lost(self, processor):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        chunks = processor.chunk_text(text, max_tokens=8)
        joined = " ".join(chunks)
        for word in ("Alpha", "gamma", "Delta", "zeta", "Eta", "iota"):
            assert word in joined


class TestExtraction:
    def test_extracts_text_from_pdf(self, processor, pdf_bytes):
        text = processor.extract_text_from_pdf(io.BytesIO(pdf_bytes))
        assert "Orion program" in text
        assert "94 percent" in text

    def test_reads_metadata(self, processor, pdf_bytes):
        metadata = processor.get_pdf_metadata(io.BytesIO(pdf_bytes))
        assert metadata["num_pages"] == 1
        assert metadata["title"] == "Orion Program Report"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Automated test fixture"

    def test_invalid_pdf_returns_error_metadata(self, processor):
        metadata = processor.get_pdf_metadata(io.BytesIO(b"not a pdf"))
        assert "error" in metadata

    def test_invalid_pdf_raises_runtime_error(self, processor):
        with pytest.raises(RuntimeError, match="Could not read PDF"):
            processor.extract_text_from_pdf(io.BytesIO(b"not a pdf"))

    def test_accepts_path_input(self, processor, tmp_path, pdf_bytes):
        path = tmp_path / "sample.pdf"
        path.write_bytes(pdf_bytes)
        assert "Orion program" in processor.extract_text_from_pdf(path)


class TestOfflineFallback:
    def test_estimate_is_used_when_tokenizer_unavailable(self, monkeypatch):
        processor = PDFProcessor()
        monkeypatch.setattr(
            "pdf_processor.tiktoken.get_encoding",
            lambda _name: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        assert processor.count_tokens("abcdefgh") == 2
        chunks = processor.chunk_text("One sentence here. Another sentence there.", max_tokens=4)
        assert chunks
