"""
Multi-format document and media parsing engine.
Provides high-fidelity extraction for:
- PDF (.pdf) documents via pypdf
- Excel (.xlsx, .xls) workbooks via pandas/openpyxl
- CSV (.csv) tabular data via pandas
- Infographic images (.png, .jpg, .jpeg, .webp) via Gemini Multimodal Vision + Pillow
- Structured JSON and plaintext/markdown
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("doc_assistant.parsers")

# Project root anchor for portable relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _format_rel_path(path: Union[str, Path]) -> str:
    """Formats a path relative to PROJECT_ROOT for portable cross-machine logging."""
    p = Path(path)
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _dataframe_to_markdown(df: Any, max_rows: int = 100) -> str:
    """
    Renders a pandas DataFrame as a clean Markdown table string
    without requiring third-party tabulate.
    """
    if df.empty:
        return "*[Empty Table]*"

    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows: List[str] = []
    display_df = df.head(max_rows)
    for _, row in display_df.iterrows():
        cells = []
        for val in row:
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                cells.append("")
            else:
                # Replace newlines inside cells to maintain markdown table formatting
                cells.append(str(val).replace("\n", " ").replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")

    table_md = "\n".join([header, separator] + rows)
    if len(df) > max_rows:
        table_md += f"\n\n*(Display truncated to first {max_rows} rows of {len(df)} total rows)*"
    return table_md


def _compute_numeric_summaries(df: Any) -> Dict[str, Any]:
    """Computes summary sums and statistics for numerical columns in a dataframe."""
    import pandas as pd
    summaries = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if not series.empty:
                summaries[str(col)] = {
                    "total_sum": float(series.sum()),
                    "mean": float(series.mean()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "count": int(series.count())
                }
    return summaries


def parse_csv(file_path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
    """
    Parses a CSV tabular file into structured Markdown with schema and statistical metrics.
    """
    import pandas as pd

    p = Path(file_path).resolve()
    rel_path = _format_rel_path(p)
    logger.info(f"Parsing CSV file '{rel_path}'...")

    try:
        # Detect delimiter and load
        df = pd.read_csv(p, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="latin1")

    row_count, col_count = df.shape
    columns = [str(c) for c in df.columns]
    numeric_stats = _compute_numeric_summaries(df)

    content_parts = [
        f"# Tabular Data: {p.stem}",
        f"**Source File:** `{rel_path}`",
        f"**Dimensions:** {row_count} rows × {col_count} columns",
        f"**Columns:** {', '.join(columns)}\n",
    ]

    if numeric_stats:
        content_parts.append("### Key Column Summaries & Totals:")
        for col_name, stats in numeric_stats.items():
            content_parts.append(
                f"- **{col_name}**: Sum = {stats['total_sum']:,.2f}, Mean = {stats['mean']:,.2f}, "
                f"Range = [{stats['min']:,.2f} to {stats['max']:,.2f}] (Count: {stats['count']})"
            )
        content_parts.append("")

    content_parts.append("### Data Table:")
    content_parts.append(_dataframe_to_markdown(df))

    metadata = {
        "source_file": rel_path,
        "format": "csv",
        "row_count": row_count,
        "column_count": col_count,
        "columns": columns,
        "numeric_summaries": numeric_stats,
    }

    return "\n".join(content_parts), metadata


def parse_excel(file_path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
    """
    Parses an Excel spreadsheet (.xlsx, .xls) across all worksheets
    into structured Markdown sections and tabular summaries.
    """
    import pandas as pd

    p = Path(file_path).resolve()
    rel_path = _format_rel_path(p)
    logger.info(f"Parsing Excel workbook '{rel_path}'...")

    all_sheet_meta = {}
    total_rows = 0

    with pd.ExcelFile(p) as excel_file:
        sheet_names = excel_file.sheet_names

        content_parts = [
            f"# Excel Workbook: {p.stem}",
            f"**Source File:** `{rel_path}`",
            f"**Worksheets ({len(sheet_names)}):** {', '.join(sheet_names)}\n",
        ]

        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            rows, cols = df.shape
            total_rows += rows
            columns = [str(c) for c in df.columns]
            numeric_stats = _compute_numeric_summaries(df)

            content_parts.append(f"## Sheet: {sheet_name}")
            content_parts.append(f"**Dimensions:** {rows} rows × {cols} columns")
            content_parts.append(f"**Columns:** {', '.join(columns)}")

            if numeric_stats:
                content_parts.append("**Summary Metrics:**")
                for col_name, stats in numeric_stats.items():
                    content_parts.append(
                        f"- **{col_name}**: Sum = {stats['total_sum']:,.2f}, Mean = {stats['mean']:,.2f} "
                        f"[Min: {stats['min']:,.2f}, Max: {stats['max']:,.2f}]"
                    )

            content_parts.append("\n" + _dataframe_to_markdown(df) + "\n")

            all_sheet_meta[sheet_name] = {
                "row_count": rows,
                "column_count": cols,
                "columns": columns,
                "numeric_summaries": numeric_stats,
            }

    metadata = {
        "source_file": rel_path,
        "format": "excel",
        "sheet_names": sheet_names,
        "total_worksheets": len(sheet_names),
        "total_rows": total_rows,
        "sheets": all_sheet_meta,
    }

    return "\n".join(content_parts), metadata


def parse_pdf(file_path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
    """
    Parses a PDF document page-by-page using pypdf, extracting document metadata
    and preserving page boundary demarcations.
    """
    import pypdf

    p = Path(file_path).resolve()
    rel_path = _format_rel_path(p)
    logger.info(f"Parsing PDF document '{rel_path}'...")

    reader = pypdf.PdfReader(str(p))
    num_pages = len(reader.pages)

    # Extract standard PDF metadata if present
    pdf_meta = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            clean_k = k.lstrip("/").lower()
            if v and isinstance(v, str):
                pdf_meta[clean_k] = v.strip()

    title = pdf_meta.get("title") or p.stem.replace("_", " ").title()

    content_parts = [
        f"# PDF Document: {title}",
        f"**Source File:** `{rel_path}`",
        f"**Total Pages:** {num_pages}",
    ]
    if pdf_meta.get("author"):
        content_parts.append(f"**Author:** {pdf_meta['author']}")
    if pdf_meta.get("subject"):
        content_parts.append(f"**Subject:** {pdf_meta['subject']}")
    content_parts.append("\n--- [Document Content Start] ---\n")

    total_chars = 0
    pages_extracted = 0

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        page_text = page.extract_text() or ""
        cleaned_text = page_text.strip()
        total_chars += len(cleaned_text)

        content_parts.append(f"--- [Page {page_num} of {num_pages}] ---")
        if cleaned_text:
            content_parts.append(cleaned_text)
            pages_extracted += 1
        else:
            content_parts.append("*[Visual or image-only page; no raw text layer detected]*")
        content_parts.append("")

    content_parts.append("--- [Document Content End] ---")

    metadata = {
        "source_file": rel_path,
        "format": "pdf",
        "title": title,
        "total_pages": num_pages,
        "pages_with_text": pages_extracted,
        "char_count": total_chars,
        "pdf_info": pdf_meta,
    }

    return "\n".join(content_parts), metadata


def _extract_text_content(content: Any) -> str:
    """Helper to cleanly extract text from string or message content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def parse_image_infographic(
    file_path: Union[str, Path],
    llm: Optional[Any] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Parses an infographic or document image (.png, .jpg, .jpeg, .webp) using
    Google Gemini Multimodal Vision, transcribing all charts, metrics, callouts,
    and tables into structured Markdown.
    """
    from PIL import Image

    p = Path(file_path).resolve()
    rel_path = _format_rel_path(p)
    logger.info(f"Parsing infographic image '{rel_path}'...")

    # Open image with Pillow to extract image properties
    with Image.open(p) as img:
        img_format = (img.format or p.suffix.lstrip(".").upper()).upper()
        width, height = img.size
        img_mode = img.mode

    # Prepare base64 image data
    with open(p, "rb") as f:
        img_bytes = f.read()
    b64_encoded = base64.b64encode(img_bytes).decode("utf-8")

    # Determine MIME type
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(p.suffix.lower(), "image/jpeg")

    # If no LLM passed, attempt to instantiate one via Gemini if API key is present
    vision_llm = llm
    if vision_llm is None:
        try:
            from dotenv import load_dotenv
            from langchain_google_genai import ChatGoogleGenerativeAI

            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                model_name = os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest"
                if "1.5" in model_name or "2.5" in model_name or "3.8" in model_name:
                    model_name = "gemini-flash-lite-latest"
                vision_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.0
                )
        except Exception as ex:
            logger.warning(f"Could not initialize Vision LLM for image parsing: {ex}")

    extracted_markdown = ""
    vision_success = False

    if vision_llm:
        from langchain_core.messages import HumanMessage

        prompt_text = (
            "You are an expert document, chart, and infographic parser. "
            "Analyze and transcribe all visual and textual elements in this infographic or image in detail.\n\n"
            "Extract and structure the following:\n"
            "1. **Title & Theme**: The main header, subtitle, topic, and overall theme.\n"
            "2. **Key Data & Metrics**: Extract every specific number, dollar amount, percentage, ratio, date, and statistic shown.\n"
            "3. **Visual Charts & Graphs**: For every chart (bar chart, pie chart, timeline, line graph, diagram), describe:\n"
            "   - Chart title and type\n"
            "   - X-axis and Y-axis labels and scales\n"
            "   - All category values, data points, percentages, and trends\n"
            "4. **Tabular Data**: Convert any tables, comparative lists, or multi-column data into clean Markdown tables.\n"
            "5. **Callouts, Notes & Citations**: Include all section headings, explanatory notes, icons with text, footnotes, and source citations.\n\n"
            "Ensure 100% factual accuracy with zero approximations. Format the entire transcription in clear, structured Markdown."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_encoded}"}
            ]
        )

        try:
            response = vision_llm.invoke([message])
            extracted_markdown = _extract_text_content(response.content)
            vision_success = True
            logger.info(f"Successfully transcribed infographic '{rel_path}' using Gemini Vision.")
        except Exception as e:
            logger.error(f"Gemini Vision extraction failed for '{rel_path}': {e}")
            extracted_markdown = (
                f"*[Vision extraction encountered an error: {str(e)}]*\n\n"
                f"Image properties: {width}x{height} pixels, format: {img_format}."
            )

    if not vision_llm and not vision_success:
        extracted_markdown = (
            f"*[Image attached: {p.name}]*\n\n"
            f"Dimensions: {width}x{height} pixels | Format: {img_format} | Mode: {img_mode}.\n"
            f"(Note: Configure GEMINI_API_KEY to enable automated multimodal visual transcription)."
        )

    content_parts = [
        f"# Infographic Analysis: {p.stem.replace('_', ' ').title()}",
        f"**Source File:** `{rel_path}`",
        f"**Image Dimensions:** {width} × {height} px ({img_format})\n",
        extracted_markdown
    ]

    metadata = {
        "source_file": rel_path,
        "format": "image",
        "image_format": img_format,
        "dimensions": f"{width}x{height}",
        "mime_type": mime_type,
        "vision_extracted": vision_success,
    }

    return "\n".join(content_parts), metadata


def parse_file(
    file_path: Union[str, Path],
    doc_type: Optional[str] = None,
    llm: Optional[Any] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Unified dispatcher for parsing any supported document or media file.
    Supported extensions: .pdf, .xlsx, .xls, .csv, .png, .jpg, .jpeg, .webp, .json, .md, .txt.

    Returns:
        Tuple[str, str, Dict[str, Any]]: (content, resolved_doc_type, metadata)
    """
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    rel_path = _format_rel_path(p)

    if ext == ".pdf":
        content, meta = parse_pdf(p)
        resolved_type = doc_type or "pdf_document"
        return content, resolved_type, meta

    elif ext in (".xlsx", ".xls"):
        content, meta = parse_excel(p)
        resolved_type = doc_type or "excel_spreadsheet"
        return content, resolved_type, meta

    elif ext == ".csv":
        content, meta = parse_csv(p)
        resolved_type = doc_type or "csv_table"
        return content, resolved_type, meta

    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        content, meta = parse_image_infographic(p, llm=llm)
        resolved_type = doc_type or "infographic_image"
        return content, resolved_type, meta

    elif ext == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        resolved_type = doc_type or data.get("doc_type", "structured_json")
        content = data.get("content", json.dumps(data, indent=2))
        meta = data.get("metadata", {})
        meta["source_file"] = rel_path
        return content, resolved_type, meta

    elif ext in (".md", ".txt"):
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        resolved_type = doc_type or "text_document"
        meta = {"source_file": rel_path}
        return content, resolved_type, meta

    else:
        # Fallback for unrecognized text files
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        resolved_type = doc_type or "external_file"
        meta = {"source_file": rel_path}
        return content, resolved_type, meta
