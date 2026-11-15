"""Streamlit UI for the PDF AI Summarizer.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from config import CHUNK_TOKEN_MAX, CHUNK_TOKEN_MIN, Settings, configure_logging
from pdf_processor import PDFProcessor
from prompts import DEFAULT_STYLE, STYLE_LABELS
from summarizer import PDFSummarizer
from utils import (
    create_summary_dataframe,
    estimate_processing_time,
    export_summary_to_text,
    format_summary_display,
    summary_stem,
    validate_api_key,
)

st.set_page_config(page_title="PDF AI Summarizer", page_icon="📄", layout="wide")

configure_logging()


def get_settings() -> Settings:
    """Load and cache settings; stop the page when configuration is invalid."""
    if "settings" not in st.session_state:
        settings = validate_api_key()
        if settings is None:
            st.stop()
        st.session_state.settings = settings
    return st.session_state.settings


def get_summarizer(settings: Settings, model: str) -> PDFSummarizer:
    """Recreate the summarizer only when the selected model changes."""
    key = f"summarizer::{model}"
    if key not in st.session_state:
        st.session_state[key] = PDFSummarizer(model=model, settings=settings)
    return st.session_state[key]


def render_sidebar(settings: Settings) -> dict[str, Any]:
    st.sidebar.header("Settings")

    model = st.sidebar.selectbox(
        "Model",
        settings.model_choices,
        help="Gemini model used for every LLM call",
    )

    summary_type = st.sidebar.selectbox(
        "Summary Type",
        list(STYLE_LABELS),
        index=list(STYLE_LABELS).index(DEFAULT_STYLE),
        format_func=lambda value: STYLE_LABELS[value],
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens per Chunk",
        CHUNK_TOKEN_MIN,
        CHUNK_TOKEN_MAX,
        value=settings.max_tokens_per_chunk,
        step=500,
        help="Smaller chunks mean more calls but tighter context",
    )

    show_analysis = st.sidebar.checkbox(
        "Show Document Analysis", value=True, help="Analyze structure and content"
    )
    show_quotes = st.sidebar.checkbox(
        "Extract Key Quotes", value=False, help="Pull important verbatim quotes"
    )

    return {
        "model": model,
        "summary_type": summary_type,
        "max_tokens": max_tokens,
        "show_analysis": show_analysis,
        "show_quotes": show_quotes,
    }


def process_pdf(
    uploaded_file,
    options: dict[str, Any],
    processor: PDFProcessor,
    summarizer: PDFSummarizer,
) -> dict[str, Any] | None:
    """Run the full pipeline and return a result payload (or None on failure)."""
    started = time.perf_counter()

    with st.spinner("Extracting text from PDF..."):
        try:
            raw_text = processor.extract_text_from_pdf(uploaded_file)
        except (RuntimeError, ValueError) as exc:
            st.error(f"Error extracting text: {exc}")
            return None
        st.success(f"Extracted {len(raw_text):,} characters")

    with st.spinner("Cleaning and chunking text..."):
        clean_text = processor.clean_text(raw_text)
        token_count = processor.count_tokens(clean_text)
        st.success(f"Cleaned text: {len(clean_text):,} characters (~{token_count:,} tokens)")
        try:
            chunks = processor.chunk_text(clean_text, options["max_tokens"])
        except ValueError as exc:
            st.error(str(exc))
            return None
        st.success(f"Created {len(chunks)} chunk(s)")
        concurrency = summarizer.settings.max_concurrency if summarizer.settings else 4
        st.info(f"Estimated processing time: {estimate_processing_time(len(chunks), concurrency)}")

    analysis = None
    if options["show_analysis"]:
        with st.spinner("Analyzing document structure..."):
            analysis = summarizer.analyze_document_structure(clean_text)

    with st.spinner(f"Generating {options['summary_type']} summaries..."):
        try:
            summary_data = summarizer.summarize_chunks(chunks, options["summary_type"])
        except Exception as exc:  # noqa: BLE001 - surface the error in the UI
            st.error(f"Error generating summaries: {exc}")
            return None

    quotes = None
    if options["show_quotes"]:
        with st.spinner("Extracting key quotes..."):
            try:
                quotes = summarizer.extract_key_quotes(clean_text)
            except Exception:  # noqa: BLE001 - optional feature
                st.warning("Could not extract key quotes")

    return {
        "file_name": uploaded_file.name,
        "summary_data": summary_data,
        "analysis": analysis,
        "quotes": quotes,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "characters": len(clean_text),
        "tokens": token_count,
        "chunks": len(chunks),
    }


def display_results(results: dict[str, Any]) -> None:
    """Render stored results, including download buttons."""
    analysis = results.get("analysis")
    if analysis:
        if analysis["status"] == "success":
            st.subheader("Document Analysis")
            st.write(analysis["analysis"])
        else:
            st.warning("Could not analyze document structure")

    if results.get("quotes"):
        st.subheader("Key Quotes")
        for index, quote in enumerate(results["quotes"], start=1):
            st.write(f'{index}. *"{quote}"*')

    st.header("Summary Results")
    summary_data = results["summary_data"]
    format_summary_display(summary_data)

    st.caption(
        f"Processed {results['characters']:,} characters in {results['chunks']} chunk(s) "
        f"in {results['elapsed_seconds']} s"
    )

    st.subheader("Export Options")
    stem = summary_stem(results["file_name"])
    summary_df = create_summary_dataframe(summary_data)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as Text",
            data=export_summary_to_text(summary_data, results["file_name"]),
            file_name=f"{stem}_summary.txt",
            mime="text/plain",
        )
    with col2:
        st.download_button(
            label="Download Statistics as CSV",
            data=summary_df.to_csv(index=False),
            file_name=f"{stem}_stats.csv",
            mime="text/csv",
        )

    with st.expander("Processing Statistics"):
        st.dataframe(summary_df, hide_index=True)
        st.metric("Average Compression Ratio", f"{summary_df['Compression Ratio'].mean():.2f}")
        success_rate = (summary_df["Status"] == "Success").mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")


def main() -> None:
    settings = get_settings()

    st.title("📄 PDF Reader + AI Summarizer")
    st.markdown(
        "Upload a PDF and get a structured summary, a document analysis and key quotes "
        "powered by Google Gemini."
    )

    options = render_sidebar(settings)
    summarizer = get_summarizer(settings, options["model"])

    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    processor: PDFProcessor = st.session_state.pdf_processor

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", help="Upload a PDF document to summarize"
    )

    # Drop stale results when the file changes or is removed.
    file_id = uploaded_file.file_id if uploaded_file else None
    if st.session_state.get("file_id") != file_id:
        st.session_state.pop("results", None)
        st.session_state.file_id = file_id

    if uploaded_file is None:
        st.info("Upload a PDF to get started. Scanned PDFs need OCR before upload.")
        return

    st.success(f"File uploaded: {uploaded_file.name}")
    with st.spinner("Reading PDF metadata..."):
        metadata = processor.get_pdf_metadata(uploaded_file)

    if "error" in metadata:
        st.warning(f"Could not read metadata: {metadata['error']}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Pages: {metadata.get('num_pages', 'Unknown')}")
            st.info(f"Title: {metadata.get('title', 'Unknown')}")
        with col2:
            st.info(f"Author: {metadata.get('author', 'Unknown')}")
            st.info(f"Subject: {metadata.get('subject', 'Unknown')}")

    if st.button("Process PDF", type="primary"):
        results = process_pdf(uploaded_file, options, processor, summarizer)
        if results:
            st.session_state.results = results

    # Render from session state so results survive reruns (e.g. download clicks).
    if "results" in st.session_state:
        display_results(st.session_state.results)


if __name__ == "__main__":
    main()
