"""Tests for the Phase 0 setup validator (no network access)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "validate_setup.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_setup", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before executing so dataclasses can resolve the module namespace.
    sys.modules["validate_setup"] = module
    spec.loader.exec_module(module)
    return module


validator = load_validator()


class TestMask:
    def test_masks_the_middle(self):
        assert validator.mask("abcdefghijklmnop") == "abcd...mnop"

    def test_short_keys_are_fully_masked(self):
        assert validator.mask("short") == "*****"


class TestExitCodes:
    def test_all_passed(self):
        report = validator.Report()
        report.add("GEMINI_API_KEY", validator.PASS, "ok")
        assert report.exit_code() == 0

    def test_warnings_do_not_fail(self):
        report = validator.Report()
        report.add("GEMINI_MODEL", validator.WARN, "defaulted")
        assert report.exit_code() == 0

    def test_config_failure_returns_one(self):
        report = validator.Report()
        report.add(".env file", validator.FAIL, "missing")
        assert report.exit_code() == 1

    def test_api_failure_returns_two(self):
        report = validator.Report()
        report.add("API key", validator.FAIL, "unauthorized")
        assert report.exit_code() == 2

    def test_skips_do_not_fail(self):
        report = validator.Report()
        report.add("Network", validator.SKIP, "offline")
        assert report.exit_code() == 0

    def test_counts_are_grouped_by_status(self):
        report = validator.Report()
        report.add("a", validator.PASS, "")
        report.add("b", validator.PASS, "")
        report.add("c", validator.FAIL, "")
        assert report.counts == {"PASS": 2, "WARN": 0, "FAIL": 1, "SKIP": 0}


class TestEnvLoading:
    def test_parses_quotes_and_ignores_comments(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VALIDATOR_TEST_KEY", raising=False)
        monkeypatch.delenv("VALIDATOR_OTHER_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# a comment\n"
            "\n"
            'VALIDATOR_TEST_KEY="quoted-value"\n'
            "VALIDATOR_OTHER_KEY=plain\n"
            "BROKEN LINE\n",
            encoding="utf-8",
        )

        validator.load_env(env_file)

        import os

        assert os.environ["VALIDATOR_TEST_KEY"] == "quoted-value"
        assert os.environ["VALIDATOR_OTHER_KEY"] == "plain"

    def test_missing_file_is_a_no_op(self, tmp_path):
        validator.load_env(tmp_path / "does-not-exist.env")


class TestHttpErrorFormatting:
    def test_formats_json_error_body(self):
        import io
        import json
        import urllib.error

        error = urllib.error.HTTPError(
            url="https://example.com",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(
                json.dumps(
                    {"error": {"status": "PERMISSION_DENIED", "message": "API key invalid"}}
                ).encode()
            ),
        )
        assert validator.describe_http_error(error) == (
            "HTTP 403 PERMISSION_DENIED: API key invalid"
        )

    def test_handles_non_json_body(self):
        import io
        import urllib.error

        error = urllib.error.HTTPError(
            url="https://example.com",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=io.BytesIO(b"<html>oops</html>"),
        )
        assert validator.describe_http_error(error) == "HTTP 500: Server Error"


def test_offline_flag_is_accepted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ." + "a" * 48)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    # Offline mode never touches the network; the exit code depends only on
    # whether a .env file exists in the project root (absent in CI).
    assert validator.main(["--offline"]) in (0, 1)
