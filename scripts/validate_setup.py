#!/usr/bin/env python
"""Phase 0 doctor: validate the Gemini API key and environment before running the app.

Checks, in order:
  1. .env file exists in the project root
  2. GEMINI_API_KEY is set and looks like a Google API key
  3. GEMINI_MODEL is configured
  4. The Gemini API host is reachable
  5. The key authenticates against ``models.list``
  6. The configured model is available to this key
  7. A tiny ``generateContent`` call succeeds (proves inference, not just auth)

Usage:
    python scripts/validate_setup.py            # full check
    python scripts/validate_setup.py --offline  # config only, no network
    python scripts/validate_setup.py --no-generate
    python scripts/validate_setup.py --json

Exit codes:
    0  everything passed (warnings allowed)
    1  configuration problem (.env / key / model name)
    2  network or API problem
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
TIMEOUT_SECONDS = 30

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
STATUS_ICON = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def counts(self) -> dict[str, int]:
        return {s: sum(c.status == s for c in self.checks) for s in (PASS, WARN, FAIL, SKIP)}

    def exit_code(self) -> int:
        if not self.failed:
            return 0
        config_names = {".env file", "GEMINI_API_KEY", "GEMINI_MODEL"}
        return 1 if any(c.status == FAIL and c.name in config_names for c in self.checks) else 2


def load_env(path: Path) -> None:
    """Load .env into os.environ, tolerating a missing python-dotenv install."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(path)
        return
    except ImportError:
        pass

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def api_request(
    path: str,
    key: str,
    payload: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Call the Gemini REST API and return the decoded JSON body."""
    query = {"key": key, **(params or {})}
    url = f"{API_ROOT}/{path}?{urllib.parse.urlencode(query)}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def describe_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8"))
        message = body.get("error", {}).get("message", "")
        status = body.get("error", {}).get("status", "")
        return f"HTTP {exc.code} {status}: {message}".strip()
    except (ValueError, OSError):
        return f"HTTP {exc.code}: {exc.reason}"


def mask(key: str) -> str:
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def check_env_file(report: Report) -> None:
    if ENV_PATH.exists():
        report.add(".env file", PASS, f"found at {ENV_PATH}")
    else:
        report.add(
            ".env file",
            FAIL,
            f"missing at {ENV_PATH} - copy .env.example to .env and add your key",
        )


def check_api_key(report: Report) -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        report.add(
            "GEMINI_API_KEY",
            FAIL,
            "not set - get a key at https://aistudio.google.com/apikey",
        )
        return None
    if len(key) < 20:
        report.add("GEMINI_API_KEY", FAIL, f"too short ({len(key)} chars) - looks truncated")
        return None
    if not (key.startswith("AIza") or key.startswith("AQ.")):
        report.add(
            "GEMINI_API_KEY",
            WARN,
            f"unusual prefix for a Gemini key ({mask(key)}) - continuing anyway",
        )
    else:
        report.add("GEMINI_API_KEY", PASS, f"set ({len(key)} chars, {mask(key)})")
    return key


def check_model_config(report: Report) -> str:
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    if not os.getenv("GEMINI_MODEL", "").strip():
        report.add("GEMINI_MODEL", WARN, f"not set - defaulting to {DEFAULT_MODEL}")
    else:
        report.add("GEMINI_MODEL", PASS, model)
    return model


def check_network(report: Report) -> None:
    request = urllib.request.Request(f"{API_ROOT}/models", method="HEAD")
    try:
        urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
        report.add("Network", PASS, "generativelanguage.googleapis.com reachable")
    except urllib.error.HTTPError:
        # Any HTTP response proves the host is reachable.
        report.add("Network", PASS, "generativelanguage.googleapis.com reachable")
    except (urllib.error.URLError, OSError) as exc:
        report.add("Network", FAIL, f"cannot reach Gemini API: {exc}")


def check_models(report: Report, key: str, model: str) -> list[str]:
    try:
        payload = api_request("models", key, params={"pageSize": 1000})
    except urllib.error.HTTPError as exc:
        report.add("API key", FAIL, describe_http_error(exc))
        return []
    except (urllib.error.URLError, OSError, ValueError) as exc:
        report.add("API key", FAIL, f"request failed: {exc}")
        return []

    names = [m.get("name", "").removeprefix("models/") for m in payload.get("models", [])]
    report.add("API key", PASS, f"authenticated - {len(names)} models available")

    if model in names:
        report.add("Model access", PASS, f"{model} is available")
    else:
        suggestions = [n for n in names if "flash" in n and "image" not in n and "tts" not in n]
        hint = ", ".join(suggestions[:5]) or "none"
        report.add(
            "Model access",
            FAIL,
            f"{model} is not available to this key (try: {hint})",
        )
    return names


def check_generation(report: Report, key: str, model: str) -> None:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with exactly: OK"}]}],
        "generationConfig": {"maxOutputTokens": 32, "temperature": 0},
    }
    started = time.perf_counter()
    try:
        response = api_request(f"models/{model}:generateContent", key, payload)
    except urllib.error.HTTPError as exc:
        report.add("Generation ping", FAIL, describe_http_error(exc))
        return
    except (urllib.error.URLError, OSError, ValueError) as exc:
        report.add("Generation ping", FAIL, f"request failed: {exc}")
        return

    elapsed_ms = (time.perf_counter() - started) * 1000
    candidates = response.get("candidates") or []
    parts = (candidates[0].get("content", {}).get("parts") or []) if candidates else []
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        report.add("Generation ping", FAIL, "model returned an empty response")
        return

    usage = response.get("usageMetadata", {})
    report.add(
        "Generation ping",
        PASS,
        f"model replied {text[:20]!r} in {elapsed_ms:.0f} ms "
        f"({usage.get('totalTokenCount', '?')} tokens)",
    )


def print_report(report: Report) -> None:
    width = max(len(c.name) for c in report.checks) + 2
    line = "-" * 74
    print(line)
    print(" Gemini API setup check")
    print(line)
    for check in report.checks:
        print(f" {STATUS_ICON[check.status]} {check.name:<{width}} {check.detail}")
    print(line)
    counts = report.counts
    print(
        f" {counts[PASS]} passed, {counts[WARN]} warning(s), "
        f"{counts[FAIL]} failed, {counts[SKIP]} skipped"
    )
    if report.failed:
        print(" Setup is NOT ready - fix the [FAIL] items above and re-run.")
    else:
        print(" Setup looks good - start the app with: streamlit run app.py")
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Gemini API setup.")
    parser.add_argument("--offline", action="store_true", help="skip all network checks")
    parser.add_argument("--no-generate", action="store_true", help="skip the generation ping")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    load_env(ENV_PATH)
    report = Report()

    check_env_file(report)
    key = check_api_key(report)
    model = check_model_config(report)

    if args.offline:
        report.add("Network", SKIP, "--offline requested")
        report.add("API key", SKIP, "--offline requested")
        report.add("Model access", SKIP, "--offline requested")
        report.add("Generation ping", SKIP, "--offline requested")
    elif key is None:
        report.add("Network", SKIP, "needs a valid GEMINI_API_KEY")
        report.add("API key", SKIP, "needs a valid GEMINI_API_KEY")
        report.add("Model access", SKIP, "needs a valid GEMINI_API_KEY")
        report.add("Generation ping", SKIP, "needs a valid GEMINI_API_KEY")
    else:
        check_network(report)
        check_models(report, key, model)
        if args.no_generate:
            report.add("Generation ping", SKIP, "--no-generate requested")
        else:
            check_generation(report, key, model)

    if args.json:
        print(
            json.dumps(
                {
                    "checks": [c.__dict__ for c in report.checks],
                    "counts": report.counts,
                    "ready": not report.failed,
                },
                indent=2,
            )
        )
    else:
        print_report(report)

    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
