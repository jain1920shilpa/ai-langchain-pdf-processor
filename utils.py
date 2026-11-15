"""Presentation, export and estimation helpers for the Streamlit UI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import ConfigError, Settings


def format_summary_display(summary_data: dict[str, Any]) -> None:
    """Render summary statistics, the final summary and per-chunk details."""
    st.subheader("Summary Statistics")
    individual = summary_data["individual_summaries"]
    successful = sum(1 for item in individual if item["ok"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", summary_data["total_chunks"])
    col2.metric("Successful Summaries", f"{successful}/{summary_data['total_chunks']}")
    col3.metric("Summary Type", summary_data["summary_type"].replace("_", " ").title())

    st.subheader("Final Summary")
    st.write(summary_data["combined_summary"])

    with st.expander("View Individual Chunk Summaries"):
        for item in individual:
            label = f"**Chunk {item['chunk_number']}**"
            if not item["ok"]:
                label += " - failed"
            st.write(label)
            st.write(item["summary"])
            st.caption(
                f"Original: {item['original_length']:,} chars | "
                f"Summary: {item['summary_length']:,} chars"
            )
            st.divider()


def export_summary_to_text(summary_data: dict[str, Any], filename: str = "") -> str:
    """Build a plain-text report for download."""
    lines = ["PDF Summary Report", "==================", ""]
    if filename:
        lines.append(f"File: {filename}")
    lines += [
        f"Summary Type: {summary_data['summary_type'].replace('_', ' ').title()}",
        f"Total Chunks Processed: {summary_data['total_chunks']}",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "=============",
        "",
        summary_data["combined_summary"],
        "",
        "INDIVIDUAL CHUNK SUMMARIES",
        "==========================",
        "",
    ]
    for item in summary_data["individual_summaries"]:
        suffix = "" if item["ok"] else " (FAILED)"
        lines += [f"Chunk {item['chunk_number']}{suffix}:", item["summary"], ""]
    return "\n".join(lines)


def create_summary_dataframe(summary_data: dict[str, Any]) -> pd.DataFrame:
    """One row per chunk with lengths, compression ratio and status."""
    return pd.DataFrame(
        [
            {
                "Chunk": item["chunk_number"],
                "Original Length": item["original_length"],
                "Summary Length": item["summary_length"],
                "Compression Ratio": (
                    round(item["summary_length"] / item["original_length"], 2)
                    if item["original_length"]
                    else 0
                ),
                "Status": "Success" if item["ok"] else "Error",
            }
            for item in summary_data["individual_summaries"]
        ]
    )


def estimate_processing_time(num_chunks: int, concurrency: int = 4) -> str:
    """Rough estimate: ~3 s per chunk with limited concurrency, plus 5 s to combine."""
    seconds = int(num_chunks * 3 / max(concurrency, 1) + 5)
    if seconds < 60:
        return f"~{seconds} seconds"
    return f"~{seconds // 60} min {seconds % 60}s"


def summary_stem(file_name: str) -> str:
    """Filename stem without a .pdf suffix, for download names."""
    return Path(file_name).stem or "document"


def validate_api_key() -> Settings | None:
    """Load settings and report configuration problems in the UI."""
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        st.error(f"{exc}")
        st.info("Get a free key at https://aistudio.google.com/apikey")
        return None
    st.sidebar.caption(f"API key loaded ({settings.masked_key})")
    return settings
