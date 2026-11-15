"""Tests for environment-driven settings."""

from __future__ import annotations

import pytest

from config import ConfigError, Settings

KEY = "AQ." + "a" * 48


@pytest.fixture()
def clean_env(monkeypatch):
    for name in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_TEMPERATURE",
        "MAX_TOKENS_PER_CHUNK",
        "MAX_CONCURRENCY",
        "MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_reads_all_values(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.7")
    monkeypatch.setenv("MAX_TOKENS_PER_CHUNK", "4000")
    monkeypatch.setenv("MAX_CONCURRENCY", "2")
    monkeypatch.setenv("MAX_RETRIES", "1")

    settings = Settings.from_env(tmp_path / "missing.env")

    assert settings.api_key == KEY
    assert settings.model == "gemini-3.5-flash"
    assert settings.temperature == 0.7
    assert settings.max_tokens_per_chunk == 4000
    assert settings.max_concurrency == 2
    assert settings.max_retries == 1


def test_defaults_are_applied(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    settings = Settings.from_env(tmp_path / "missing.env")
    assert settings.model == "gemini-3.1-flash-lite"
    assert settings.temperature == 0.3
    assert settings.max_tokens_per_chunk == 8000
    assert settings.max_concurrency == 4


def test_missing_key_raises(clean_env, tmp_path):
    with pytest.raises(ConfigError, match="GEMINI_API_KEY is not set"):
        Settings.from_env(tmp_path / "missing.env")


def test_truncated_key_raises(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "short")
    with pytest.raises(ConfigError, match="truncated"):
        Settings.from_env(tmp_path / "missing.env")


def test_non_numeric_value_raises(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_TEMPERATURE", "hot")
    with pytest.raises(ConfigError, match="must be a number"):
        Settings.from_env(tmp_path / "missing.env")


def test_out_of_range_value_raises(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("MAX_CONCURRENCY", "99")
    with pytest.raises(ConfigError, match="between 1 and 16"):
        Settings.from_env(tmp_path / "missing.env")


def test_masked_key_hides_the_secret():
    settings = Settings(api_key=KEY)
    assert settings.masked_key == "AQ.a...aaaa"
    assert KEY not in settings.masked_key


def test_configured_model_is_first_choice():
    settings = Settings(api_key=KEY, model="custom-model")
    assert settings.model_choices[0] == "custom-model"
    assert len(settings.model_choices) == len(set(settings.model_choices))
