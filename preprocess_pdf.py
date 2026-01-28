"""
Preprocess PDF articles for semantic search.

This module extracts text from PDF files, cleans it, parses metadata,
and saves cleaned articles in JSONL format for downstream processing.

Workflow:
1. Extract raw text from each PDF
2. Clean text (fix hyphenation, remove boilerplate, normalize whitespace)
3. Parse metadata (title, source, published date, topic, URL)
4. Filter articles by minimum word count
5. Save cleaned articles to JSONL and individual text files

Author: Semantic Search Engine Project
"""

import os
import re
import json
from typing import Dict, List, Tuple
from pypdf import PdfReader

# Configuration
PDF_DIR = r"C:\Users\tcame\OneDrive\Desktop\MSBA Winter 2025\NLP\Semantic Search Engine\Semantic_Search_Engine\pdf_articles_fulltext"
OUT_JSONL = "clean_articles.jsonl"  # Output JSONL file with cleaned articles
OUT_TXT_DIR = "clean_txt"  # Directory for individual text files
MIN_WORDS = 200  # Minimum word count threshold for quality filtering


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from all pages of a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Concatenated text from all pages, separated by newlines
    """
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


def basic_clean(text: str) -> str:
    """
    Clean and normalize extracted PDF text.
    
    Performs the following operations:
    - Fixes hyphenated line breaks (e.g., "exam-\nple" -> "example")
    - Normalizes newlines and whitespace
    - Removes common boilerplate text (subscribe, cookies, etc.)
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""

    # Fix hyphenated line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert newlines to spaces (keeps paragraphs reasonably)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)          # collapse many blank lines
    text = re.sub(r"[ \t]+", " ", text)             # collapse spaces/tabs
    text = re.sub(r"\n +", "\n", text)              # trim line-leading spaces

    # Remove common boilerplate lines (customize as you see patterns)
    boiler = [
        r"subscribe", r"sign up", r"cookie", r"privacy policy",
        r"terms of service", r"advertisement", r"all rights reserved",
        r"read more", r"share this", r"follow us"
    ]
    lines = []
    for line in text.split("\n"):
        l = line.strip()
        if not l:
            continue
        if any(re.search(pat, l, re.IGNORECASE) for pat in boiler):
            continue
        lines.append(l)

    cleaned = "\n".join(lines)

    # Final whitespace cleanup
    cleaned = cleaned.strip()
    return cleaned


def parse_metadata_from_pdf_text(text: str) -> Dict[str, str]:
    """
    Your PDFs have a header like:
      Title: ...
      Source: ...
      Published: ...
      Topic: ...
      URL: ...
    We'll extract those fields if present.
    """
    meta = {}
    patterns = {
        "title": r"^Title:\s*(.*)$",
        "source": r"^Source:\s*(.*)$",
        "published": r"^Published:\s*(.*)$",
        "topic": r"^Topic:\s*(.*)$",
        "url": r"^URL:\s*(.*)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            meta[key] = m.group(1).strip()

    return meta


def strip_header_blocks(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Remove the metadata header and section labels like 'Description:' / 'Extracted Full Text:'
    while preserving content.
    """
    meta = parse_metadata_from_pdf_text(text)

    # Remove known labels
    text = re.sub(r"^Title:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Source:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Published:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Topic:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^URL:.*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"^Description:\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Content:\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Extracted Full Text:\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text, meta


def main():
    """
    Main execution function.
    
    Processes all PDF files in the configured directory:
    - Extracts and cleans text
    - Parses metadata
    - Filters by minimum word count
    - Saves to JSONL and individual text files
    """
    os.makedirs(OUT_TXT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    kept = 0
    skipped = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for fname in pdf_files:
            path = os.path.join(PDF_DIR, fname)

            raw = extract_text_from_pdf(path)
            raw = raw.strip()
            if not raw:
                skipped += 1
                continue

            raw_clean = basic_clean(raw)
            body, meta = strip_header_blocks(raw_clean)

            # Quality filter
            word_count = len(body.split())
            if word_count < MIN_WORDS:
                skipped += 1
                continue

            record = {
                "file": fname,
                "word_count": word_count,
                **meta,
                "text": body
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Also save .txt
            txt_name = os.path.splitext(fname)[0] + ".txt"
            with open(os.path.join(OUT_TXT_DIR, txt_name), "w", encoding="utf-8") as tf:
                tf.write(body)

            kept += 1

    print(f"Done. Kept: {kept} | Skipped: {skipped}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"TXT folder: {OUT_TXT_DIR}")


if __name__ == "__main__":
    main()
