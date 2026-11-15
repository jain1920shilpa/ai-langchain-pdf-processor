"""Tests for the summarization chains, using a fake chat model (no API calls)."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from summarizer import PDFSummarizer, message_text


def make_summarizer(responses: list[str]) -> PDFSummarizer:
    return PDFSummarizer(llm=FakeListChatModel(responses=responses))


def raising(_payload):
    raise RuntimeError("boom")


class TestMessageText:
    def test_plain_string_content(self):
        assert message_text(AIMessage(content="hello")) == "hello"

    def test_text_blocks_are_joined(self):
        message = AIMessage(
            content=[
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        )
        assert message_text(message) == "first\nsecond"

    def test_non_text_blocks_are_ignored(self):
        message = AIMessage(
            content=[
                {"type": "text", "text": "caption"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            ]
        )
        assert message_text(message) == "caption"

    def test_empty_content(self):
        assert message_text(AIMessage(content=[])) == ""


class TestSummarizeText:
    def test_returns_model_text(self):
        summarizer = make_summarizer(["A concise summary."])
        assert summarizer.summarize_text("Some text", "brief") == "A concise summary."

    def test_unknown_style_falls_back_to_detailed(self):
        summarizer = make_summarizer(["Fallback summary."])
        assert summarizer.summarize_text("Some text", "not-a-style") == "Fallback summary."


class TestSummarizeChunks:
    def test_maps_chunks_then_combines(self):
        summarizer = make_summarizer(["S1", "S2", "COMBINED"])
        result = summarizer.summarize_chunks(["chunk one", "chunk two"], max_concurrency=1)

        assert result["total_chunks"] == 2
        assert result["summary_type"] == "detailed"
        assert [item["summary"] for item in result["individual_summaries"]] == ["S1", "S2"]
        assert all(item["ok"] for item in result["individual_summaries"])
        assert result["combined_summary"] == "COMBINED"

    def test_records_lengths(self):
        summarizer = make_summarizer(["short", "short", "COMBINED"])
        result = summarizer.summarize_chunks(["a" * 100, "b" * 50], max_concurrency=1)
        assert result["individual_summaries"][0]["original_length"] == 100
        assert result["individual_summaries"][0]["summary_length"] == 5

    def test_empty_chunk_list_raises(self):
        summarizer = make_summarizer(["unused"])
        with pytest.raises(ValueError, match="No chunks"):
            summarizer.summarize_chunks([])

    def test_failed_chunk_is_isolated(self):
        summarizer = make_summarizer(["unused"])

        def chunk_step(payload):
            if "FAIL" in payload["text"]:
                raise RuntimeError("chunk exploded")
            return "ok summary"

        summarizer.chunk_chain = RunnableLambda(chunk_step)
        summarizer.combine_chain = RunnableLambda(lambda _payload: "COMBINED")

        result = summarizer.summarize_chunks(
            ["good chunk", "FAIL chunk", "another good chunk"], max_concurrency=1
        )
        first, second, third = result["individual_summaries"]
        assert first["ok"] is True
        assert second["ok"] is False
        assert "Error processing chunk" in second["summary"]
        assert third["ok"] is True
        assert result["combined_summary"] == "COMBINED"


class TestCombine:
    def test_single_summary_is_returned_as_is(self):
        summarizer = make_summarizer(["unused"])
        payload = [{"summary": "only one", "ok": True}]
        assert summarizer.combine_summaries(payload) == "only one"

    def test_no_valid_summaries(self):
        summarizer = make_summarizer(["unused"])
        payload = [{"summary": "failed", "ok": False}]
        assert summarizer.combine_summaries(payload) == "No valid summaries were generated."

    def test_falls_back_when_combine_call_fails(self):
        summarizer = make_summarizer(["unused"])
        summarizer.combine_chain = RunnableLambda(raising)
        payload = [{"summary": "one", "ok": True}, {"summary": "two", "ok": True}]
        result = summarizer.combine_summaries(payload)
        assert result.startswith("Combined Summary:")
        assert "Section 1: one" in result
        assert "Section 2: two" in result


class TestAnalysis:
    def test_success(self):
        summarizer = make_summarizer(["Document type: report"])
        result = summarizer.analyze_document_structure("text")
        assert result == {"analysis": "Document type: report", "status": "success"}

    def test_failure_is_reported_not_raised(self):
        summarizer = make_summarizer(["unused"])
        summarizer.analysis_chain = RunnableLambda(raising)
        result = summarizer.analyze_document_structure("text")
        assert result["status"] == "error"
        assert "boom" in result["analysis"]


class TestQuotes:
    def test_strips_numbering_bullets_and_quotes(self):
        summarizer = make_summarizer(
            ['1. "First quote"\n- Second quote\n\u2022 Third quote\n\n']
        )
        assert summarizer.extract_key_quotes("text") == [
            "First quote",
            "Second quote",
            "Third quote",
        ]

    def test_limits_to_ten_quotes(self):
        summarizer = make_summarizer(["\n".join(f"quote {i}" for i in range(20))])
        assert len(summarizer.extract_key_quotes("text")) == 10
