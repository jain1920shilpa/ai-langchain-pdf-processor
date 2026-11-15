"""PDF text extraction, cleaning, token counting and chunking.

The pipeline is deliberately dependency-light: pypdf reads the document and
tiktoken measures it. Chunking happens on sentence boundaries so the LLM never
sees a summary request that starts or ends mid-sentence, with a token-level
hard split only as a fallback for pathological input.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import BinaryIO

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = "cl100k_base"
DEFAULT_MAX_TOKENS = 8000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MULTI_SPACE = re.compile(r"[ \t]+")
_SINGLE_NEWLINE_PADDING = re.compile(r" ?\n ?")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


class PDFProcessor:
    """Turn an uploaded PDF into clean, token-bounded chunks."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        self.encoding_name = encoding_name
        self._encoding: tiktoken.Encoding | None = None

    # ------------------------------------------------------------------ tokens
    @property
    def encoding(self) -> tiktoken.Encoding | None:
        """Lazily load the tokenizer; ``None`` means we fall back to an estimate.

        tiktoken downloads its vocabulary on first use. In an offline environment
        we still want the app to work, so the fallback is a ~4-chars-per-token
        approximation rather than a hard failure.
        """
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding(self.encoding_name)
            except Exception as exc:  # noqa: BLE001 - network or cache problems
                logger.warning(
                    "tiktoken encoding %r unavailable (%s); using a 4-chars/token estimate",
                    self.encoding_name,
                    exc,
                )
                return None
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """Number of tokens in ``text`` (approximate when tiktoken is offline)."""
        if not text:
            return 0
        encoding = self.encoding
        if encoding is None:
            return max(1, len(text) // 4)
        return len(encoding.encode(text, disallowed_special=()))

    def _encode(self, text: str) -> list[int] | None:
        encoding = self.encoding
        if encoding is None:
            return None
        return encoding.encode(text, disallowed_special=())

    def _decode(self, tokens: list[int]) -> str:
        encoding = self.encoding
        if encoding is None:
            raise RuntimeError("cannot decode tokens without tiktoken")
        return encoding.decode(tokens)

    # --------------------------------------------------------------------- pdf
    def extract_text_from_pdf(self, pdf_file: str | Path | BinaryIO) -> str:
        """Return the text of every page, skipping pages that fail to parse."""
        try:
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)  # type: ignore[union-attr]
            reader = PdfReader(pdf_file)
            if reader.is_encrypted:
                raise ValueError("PDF is encrypted; remove the password and retry")
            pages: list[str] = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:  # noqa: BLE001 - one bad page must not abort
                    logger.warning("Skipping page %d: %s", page_number, exc)
            return "\n\n".join(pages)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - pypdf raises many types
            raise RuntimeError(f"Could not read PDF: {exc}") from exc

    def get_pdf_metadata(self, pdf_file: str | Path | BinaryIO) -> dict:
        """Return page count and document metadata, or ``{"error": ...}``."""
        try:
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)  # type: ignore[union-attr]
            reader = PdfReader(pdf_file)
            meta = reader.metadata or {}

            def field(key: str) -> str:
                value = meta.get(key)
                return str(value).strip() if value else "Unknown"

            return {
                "num_pages": len(reader.pages),
                "title": field("/Title"),
                "author": field("/Author"),
                "subject": field("/Subject"),
            }
        except Exception as exc:  # noqa: BLE001 - metadata is best effort
            return {"error": str(exc)}

    # ------------------------------------------------------------------- clean
    @staticmethod
    def clean_text(text: str) -> str:
        """Drop control characters and normalise whitespace without losing structure."""
        text = _CONTROL_CHARS.sub("", text)
        text = _MULTI_SPACE.sub(" ", text)
        text = _SINGLE_NEWLINE_PADDING.sub("\n", text)
        text = _EXCESS_NEWLINES.sub("\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------ chunks
    def chunk_text(self, text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[str]:
        """Split ``text`` into chunks of at most ``max_tokens``.

        Sentences are kept intact where possible. A single sentence longer than
        the budget is split on token boundaries.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        text = text.strip()
        if not text:
            raise ValueError("No extractable text - the PDF may be scanned images")

        pieces: list[tuple[str, int]] = []
        for sentence in _SENTENCE_BOUNDARY.split(text):
            if not sentence:
                continue
            tokens = self._encode(sentence)
            if tokens is None:
                # Offline fallback: split on character budget.
                char_budget = max_tokens * 4
                for start in range(0, len(sentence), char_budget):
                    piece = sentence[start : start + char_budget]
                    pieces.append((piece, self.count_tokens(piece)))
            elif len(tokens) > max_tokens:
                for start in range(0, len(tokens), max_tokens):
                    piece_tokens = tokens[start : start + max_tokens]
                    pieces.append((self._decode(piece_tokens), len(piece_tokens)))
            else:
                pieces.append((sentence, len(tokens)))

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece, token_count in pieces:
            if current and current_tokens + token_count + 1 > max_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += token_count + 1  # +1 approximates the joining space
        if current:
            chunks.append(" ".join(current))
        return chunks
