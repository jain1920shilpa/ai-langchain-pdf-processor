# Walkthrough

A tour of how the app is built and how a PDF travels through it.

## 1. Request lifecycle

```
User uploads PDF
      │
      ▼
app.py: get_settings() ── config.Settings.from_env() ── .env
      │                         └─ missing/invalid key → stop with a clear error
      ▼
pdf_processor.extract_text_from_pdf()      pypdf, page-by-page, skips bad pages
      │
      ▼
pdf_processor.clean_text()                 control chars, whitespace, paragraph breaks
      │
      ▼
pdf_processor.chunk_text(max_tokens)       sentence-aware, hard split as fallback
      │
      ├─► summarizer.analyze_document_structure()   optional, first 4k chars
      │
      ▼
summarizer.summarize_chunks()              map: batch(chunk_chain, max_concurrency)
      │
      ▼
summarizer.combine_summaries()             reduce: COMBINE prompt
      │
      ├─► summarizer.extract_key_quotes()           optional, first 6k chars
      ▼
utils.format_summary_display()             metrics, final summary, chunk details
utils.export_summary_to_text()             .txt download
utils.create_summary_dataframe()           .csv download + compression stats
```

## 2. Module responsibilities

| Module | Owns | Does not own |
|---|---|---|
| `config.py` | Env parsing, validation, model list, logging | Any UI or LLM calls |
| `pdf_processor.py` | Bytes → clean text → chunks; metadata | Prompts, LLM, UI |
| `prompts.py` | System persona, styles, templates | Chain wiring |
| `summarizer.py` | Chain construction, retries, map-reduce, parsing | File I/O, UI |
| `utils.py` | Rendering helpers, exports, estimates | Business logic |
| `app.py` | Session state, widgets, orchestration | Prompt or PDF internals |
| `scripts/validate_setup.py` | Pre-flight checks | App runtime |

The dependency direction is one-way: `app → utils/summarizer/pdf_processor →
prompts/config`. Nothing in the lower layers imports Streamlit, which keeps the
core testable without a browser.

## 3. Why map-reduce

Gemini models accept large contexts, but a single call over a 200-page document
is slow, expensive to retry and produces uneven summaries. Chunking gives:

- bounded request size and predictable cost per chunk
- partial progress: one failed chunk does not lose the whole document
- concurrency: `max_concurrency` chunks in flight at once

The trade-off is repetition between chunk summaries, which the `COMBINE` prompt
explicitly removes.

## 4. Failure handling

| Failure | Behaviour |
|---|---|
| Page cannot be parsed | Warning logged, page skipped |
| PDF is encrypted | `ValueError` with a clear message |
| No extractable text | `ValueError` suggesting OCR |
| One chunk call fails | Recorded as failed, other chunks continue |
| Combine call fails | Falls back to concatenated sections |
| Analysis/quotes fail | Non-fatal warning; summary still shown |
| 429 / 5xx | Retried with exponential backoff (`max_retries`) |

## 5. Session state

Streamlit reruns the whole script on every interaction (button, download click,
widget change). Three keys keep that cheap and consistent:

- `settings` — loaded once per session
- `summarizer` — keyed by model, recreated only when the model changes
- `results` — cleared when the uploaded file changes, rendered on rerun so
  downloads do not re-run the pipeline

## 6. Testing strategy

| Layer | Approach |
|---|---|
| `config` | Env monkeypatching, invalid values |
| `pdf_processor` | In-memory PDF built with fpdf2; chunk budget and loss checks |
| `summarizer` | `FakeListChatModel` and `RunnableLambda` stubs; no API calls |
| `utils` | Pure functions: exports, dataframe, estimates |
| `validate_setup` | Module loaded by path; offline flag; exit codes |
| Integration | Opt-in live calls via `RUN_INTEGRATION=1` |

## 7. Build phases

| Phase | Output | Gate |
|---|---|---|
| 0 | Env files, validator, pinned deps | Validator exits 0, `pip check` clean |
| 1 | `config.py`, `pyproject.toml` | `ruff check` passes |
| 2 | `pdf_processor.py` | PDF unit tests pass |
| 3 | `prompts.py`, `summarizer.py` | Mocked-LLM tests pass |
| 4 | `app.py` | App serves HTTP 200 |
| 5 | `utils.py` | Export tests pass |
| 6 | Tests, docs, CI | Full suite green |
| 7 | Git + GitHub | Remote pushed, CI green |

## 8. Extending the app

**New summary style** — add to `STYLES` and `STYLE_LABELS` in `prompts.py`.

**Different provider** — `PDFSummarizer` accepts any `BaseChatModel` through
`llm=`. Swap `ChatGoogleGenerativeAI` for another chat model; chains stay intact.

**New output format** — add a builder in `utils.py` and a `st.download_button`
in `display_results()`.

**Chat with the document** — add a retrieval step before `summarize_text`; the
chunking layer already gives you retrieval units.
