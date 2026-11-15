# 📄 PDF AI Summarizer

A Streamlit app that summarizes PDF documents with **Google Gemini** via **LangChain**.
Upload a PDF, pick a summary style, and get a structured summary, a document
analysis and optional key quotes.

This project mirrors the feature set of
[pdfsummarizer-gemini](https://github.com/niti007/pdfsummarizer-gemini) and adds a
typed config layer, a Phase-0 setup validator, a pytest suite, linting and CI.

## Features

- Extracts and cleans text from a PDF, then splits it into token-bounded chunks
- Map-reduce summarization: each chunk is summarized, then merged into one result
- Four summary styles:
  - **brief** — 2–3 sentence overview
  - **detailed** — key points, arguments and conclusions
  - **bullet_points** — scannable bullet list
  - **executive** — business-focused insights and recommendations
- Optional document-structure analysis and key-quote extraction
- Exports the summary as text and per-chunk statistics as CSV
- Resilient by design: per-chunk failure isolation, retries with backoff, bounded concurrency

## How it works

```
PDF → extract text → clean → chunk (by tokens)
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
             summary 1       summary 2   ...  summary N      (map)
                  └───────────────┼───────────────┘
                                  ▼
                        combine → final summary             (reduce)
```

## Requirements

- Python 3.12 or newer (developed on 3.13 x64)
- A Gemini API key from <https://aistudio.google.com/apikey>

> **Windows on ARM note:** native ARM64 Python cannot run this stack because
> `pyarrow` publishes no Windows ARM64 wheels. Use an x64 Python 3.13 install
> (it runs fine under Windows emulation). See *Environment constraints* below.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env      # then add your key
```

Validate the environment **before** launching the app:

```powershell
python scripts/validate_setup.py
```

The validator checks the `.env` file, key format, network reachability, model
access and a live generation ping, then exits non-zero if anything is wrong.

Useful flags: `--offline` (config only), `--no-generate`, `--json`.

## Run

```powershell
python -m streamlit run app.py
```

Open <http://localhost:8501>.

## Project structure

```
app.py                       Streamlit UI (settings, upload, results, downloads)
config.py                    Typed settings loaded from .env
pdf_processor.py             PDF extraction, cleaning, token counting, chunking
summarizer.py                Gemini chains: chunk summaries, combine, analysis, quotes
prompts.py                   All ChatPromptTemplates and style instructions
utils.py                     Display, export, statistics and key helpers
scripts/validate_setup.py    Phase-0 environment and API-key validator
tests/                       pytest suite (unit, mocked LLM, opt-in integration)
.github/workflows/ci.yml     Lint + test on every push
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required. Get one at aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Model used for all calls |
| `GEMINI_TEMPERATURE` | `0.3` | Sampling temperature (0–2) |
| `MAX_TOKENS_PER_CHUNK` | `8000` | Initial chunk size (2000–10000) |
| `MAX_CONCURRENCY` | `4` | Parallel chunk requests (1–16) |
| `MAX_RETRIES` | `3` | Retries per LLM call |

The sidebar can override the model, summary style and chunk size per run.

## Prompt templates

All prompts live in `prompts.py` as LangChain `ChatPromptTemplate`s
(system + human message):

- `STYLES` — instruction text for each summary type, injected as `{style}`
- `CHUNK_SUMMARY` — per-chunk summary (`{style}`, `{text}`)
- `COMBINE` — merges chunk summaries into the final summary
- `ANALYSIS` — document structure analysis
- `QUOTES` — key quote extraction

To add a style, add an entry to `STYLES` and `STYLE_LABELS` in `prompts.py`; the
sidebar picks it up automatically.

## Testing

```powershell
python -m pytest                 # unit + mocked-LLM tests, no API calls
python -m ruff check .           # lint
$env:RUN_INTEGRATION="1"; python -m pytest -m integration   # live API calls
```

The integration tests make real Gemini requests and consume a small amount of quota.

## Troubleshooting

- **"No extractable text"** — the PDF is probably scanned images. Run OCR first.
- **429 / rate limit** — wait a minute, use a smaller document, or lower `MAX_CONCURRENCY`.
- **Model not available / 404** — set `GEMINI_MODEL` to a model your key can access;
  `python scripts/validate_setup.py` lists suggestions.
- **API key not found** — ensure `.env` sits next to `app.py` and contains `GEMINI_API_KEY`.
- **`DLL load failed ... Application Control policy has blocked this file`** — Windows
  Smart App Control is blocking an unsigned DLL. See below.

## Environment constraints

This repository was built on a Windows 11 ARM64 machine with Smart App Control
enabled, which produced two constraints that are already handled:

1. **x64 Python required.** `pyarrow` has no Windows ARM64 wheels, so the venv is
   built from an x64 Python 3.13 install. On x64 Windows or Linux this is a non-issue.
2. **Pinned native dependencies.** Smart App Control blocks newer unsigned builds:
   `pyarrow==21.0.0` and `xxhash==3.5.0` are pinned in `requirements.txt` because
   the latest versions are blocked by policy. Both satisfy Streamlit's and
   LangSmith's version ranges.
3. **Launcher stubs are blocked.** `streamlit.exe` in `.venv\Scripts` may be
   blocked, so run the app as `python -m streamlit run app.py`.

If your machine has Smart App Control off, you can relax the pins.

## Security

`.env` is git-ignored and never committed. The setup validator only ever prints a
masked key (`AQ.A...YvWw`). If a key is exposed, rotate it in Google AI Studio.
