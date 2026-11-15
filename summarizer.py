"""Gemini-backed summarization chains (map-reduce over document chunks)."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Settings
from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, DEFAULT_STYLE, QUOTES, style_instruction

logger = logging.getLogger(__name__)

ANALYSIS_CHARS = 4000
QUOTES_CHARS = 6000
MAX_QUOTES = 10


def message_text(message: BaseMessage) -> str:
    """Extract plain text from a response whose content may be a list of blocks.

    Gemini 3 returns multi-part content, so text blocks are joined with newlines.
    Reading ``content`` directly avoids the deprecated ``.text()`` accessor.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(part for part in parts if part).strip()
        return joined

    return str(content).strip() if content is not None else ""


class PDFSummarizer:
    """Map-reduce summarizer with optional analysis and quote extraction."""

    def __init__(
        self,
        model: str | None = None,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        if llm is None:
            settings = settings or Settings.from_env()
            self.settings = settings
            self.model = model or settings.model
            llm = self._build_llm(settings, self.model)
        else:
            self.settings = settings
            self.model = model or getattr(llm, "model_name", None) or "injected-llm"

        self.llm = llm
        self._retrying_llm = llm.with_retry(
            stop_after_attempt=settings.max_retries if settings else 3,
            wait_exponential_jitter=True,
        )
        parser = RunnableLambda(message_text)
        self.chunk_chain = CHUNK_SUMMARY | self._retrying_llm | parser
        self.combine_chain = COMBINE | self._retrying_llm | parser
        self.analysis_chain = ANALYSIS | self._retrying_llm | parser
        self.quotes_chain = QUOTES | self._retrying_llm | parser

    @staticmethod
    def _build_llm(settings: Settings, model: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=settings.temperature,
            google_api_key=settings.api_key,
        )

    # ---------------------------------------------------------------- summarize
    def summarize_text(self, text: str, summary_type: str = DEFAULT_STYLE) -> str:
        """Summarize a single piece of text (one chunk, or a short document)."""
        return self.chunk_chain.invoke(
            {"style": style_instruction(summary_type), "text": text}
        )

    def summarize_chunks(
        self,
        chunks: list[str],
        summary_type: str = DEFAULT_STYLE,
        max_concurrency: int | None = None,
    ) -> dict[str, Any]:
        """Summarize every chunk, then merge the results into one final summary.

        Chunk failures are isolated: a failed chunk is recorded and the run
        continues with the chunks that succeeded.
        """
        if not chunks:
            raise ValueError("No chunks to summarize")

        style = style_instruction(summary_type)
        inputs = [{"style": style, "text": chunk} for chunk in chunks]
        concurrency = max_concurrency or (
            self.settings.max_concurrency if self.settings else 4
        )
        results = self.chunk_chain.batch(
            inputs,
            config={"max_concurrency": concurrency},
            return_exceptions=True,
        )

        summaries: list[dict[str, Any]] = []
        for index, (chunk, result) in enumerate(zip(chunks, results, strict=True), start=1):
            ok = not isinstance(result, Exception)
            if not ok:
                logger.warning("Chunk %d failed: %s", index, result)
            summary = str(result) if ok else f"Error processing chunk: {result}"
            summaries.append(
                {
                    "chunk_number": index,
                    "summary": summary,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(summary) if ok else 0,
                }
            )

        return {
            "individual_summaries": summaries,
            "combined_summary": self.combine_summaries(summaries, summary_type),
            "total_chunks": len(chunks),
            "summary_type": summary_type,
        }

    def combine_summaries(
        self,
        chunk_summaries: list[dict[str, Any]],
        summary_type: str = DEFAULT_STYLE,
    ) -> str:
        """Merge per-chunk summaries into one; degrade gracefully on failure."""
        valid = [item["summary"] for item in chunk_summaries if item.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(
            f"Section {index}: {summary}" for index, summary in enumerate(valid, start=1)
        )
        try:
            return self.combine_chain.invoke(
                {"style": style_instruction(summary_type), "summaries": sections}
            )
        except Exception as exc:  # noqa: BLE001 - fall back to concatenation
            logger.warning("Combine step failed (%s); returning concatenated sections", exc)
            return f"Combined Summary:\n\n{sections}"

    # ----------------------------------------------------------------- analysis
    def analyze_document_structure(self, text: str) -> dict[str, str]:
        """Best-effort structural analysis of the document's opening section."""
        try:
            analysis = self.analysis_chain.invoke({"text": text[:ANALYSIS_CHARS]})
            return {"analysis": analysis, "status": "success"}
        except Exception as exc:  # noqa: BLE001 - optional feature
            logger.warning("Document analysis failed: %s", exc)
            return {"analysis": f"Error analyzing document: {exc}", "status": "error"}

    def extract_key_quotes(self, text: str) -> list[str]:
        """Return up to MAX_QUOTES verbatim quotes, stripped of list markers."""
        response = self.quotes_chain.invoke({"text": text[:QUOTES_CHARS]})
        lines = (
            re.sub(r"^\s*(\d+[.)]|[-*\u2022])\s*", "", line).strip(' "\u201c\u201d')
            for line in response.splitlines()
        )
        return [line for line in lines if line][:MAX_QUOTES]
