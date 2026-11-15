"""Application settings, loaded from environment variables.

Keeping configuration in one typed object means the UI, the LLM layer and the
tests all read the same values, and a bad value fails fast with a clear message
instead of surfacing deep inside a LangChain call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS_PER_CHUNK = 8000
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 3

# Models offered in the sidebar. The configured model is always prepended, so a
# custom GEMINI_MODEL still shows up even when it is not in this list.
KNOWN_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
)

CHUNK_TOKEN_MIN = 2000
CHUNK_TOKEN_MAX = 10000


class ConfigError(RuntimeError):
    """Raised when the environment is missing or contains an invalid value."""


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_number(name: str, default: float, cast: type, minimum: float, maximum: float):
    raw = os.getenv(name, "").strip()
    if not raw:
        return cast(default)
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration."""

    api_key: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS_PER_CHUNK
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES

    @classmethod
    def from_env(cls, env_path: Path | str = ENV_PATH) -> Settings:
        """Load .env (if present) and build a validated Settings object."""
        load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key "
                "from https://aistudio.google.com/apikey"
            )
        if len(api_key) < 20:
            raise ConfigError("GEMINI_API_KEY looks truncated; generate a new key")
        return cls(
            api_key=api_key,
            model=_env_str("GEMINI_MODEL", DEFAULT_MODEL),
            temperature=_env_number("GEMINI_TEMPERATURE", DEFAULT_TEMPERATURE, float, 0.0, 2.0),
            max_tokens_per_chunk=_env_number(
                "MAX_TOKENS_PER_CHUNK",
                DEFAULT_MAX_TOKENS_PER_CHUNK,
                int,
                CHUNK_TOKEN_MIN,
                CHUNK_TOKEN_MAX,
            ),
            max_concurrency=_env_number("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, int, 1, 16),
            max_retries=_env_number("MAX_RETRIES", DEFAULT_MAX_RETRIES, int, 0, 10),
        )

    @property
    def masked_key(self) -> str:
        """Key preview safe to show in the UI or logs."""
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    @property
    def model_choices(self) -> list[str]:
        """Configured model first, then the known models, de-duplicated."""
        choices = [self.model]
        choices.extend(m for m in KNOWN_MODELS if m != self.model)
        return choices


def configure_logging(level: int = logging.INFO) -> None:
    """Send application logs to stderr with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
