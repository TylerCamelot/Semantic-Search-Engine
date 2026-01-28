"""
Download news articles from NewsAPI and save them as PDFs.

This module downloads articles from NewsAPI across multiple topics, extracts full text
using newspaper3k, validates text quality, and saves articles as multi-page PDFs with
proper word wrapping and formatting.

Workflow:
1. Fetch articles from NewsAPI for each topic
2. Extract full text from article URLs using newspaper3k
3. Validate text quality (length, word count, etc.)
4. Save articles as formatted PDFs with metadata

Author: Semantic Search Engine Project
"""

import os
import re
import time
import json
import hashlib
from typing import Dict, Tuple, Optional, List

import requests
from requests.exceptions import RequestException

from newspaper import Article
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# =========================
# CONFIG
# =========================

NEWSAPI_KEY = "2d2036aef9874af19184890cacb0f817"  # <-- put your key here

BASE_URL = "https://newsapi.org/v2/everything"
LANGUAGE = "en"
SORT_BY = "publishedAt"

OUTPUT_DIR = "pdf_articles_fulltext"
TARGET_PDFS = 1000

# topic -> (query, desired_count)
TOPICS: Dict[str, Tuple[str, int]] = {
    "technology": ('technology OR tech OR AI OR "artificial intelligence"', 125),
    "business": ('business OR economy OR finance OR markets', 125),
    "politics": ('politics OR government OR election OR policy', 125),
    "world": ('world OR international OR global', 125),
    "science": ('science OR research OR climate OR space', 125),
    "health": ('health OR medicine OR healthcare', 125),
    "sports": ('sports OR NBA OR NFL OR soccer', 125),
    "entertainment": ('entertainment OR movies OR music OR television', 125),
}

NEWSAPI_PAGE_SIZE = 100  # max per request
NEWSAPI_MAX_PAGES = 10   # safety cap; increase if needed

# Full-text quality gates (tune as you like)
MIN_EXTRACTED_CHARS = 1500     # require at least this many chars from newspaper3k
MIN_EXTRACTED_WORDS = 250      # require at least this many words
MIN_EXTRACTED_TO_NEWSAPI_RATIO = 3.0  # newspaper text should be significantly longer than API snippet

# Timeouts / politeness
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS_SEC = 0.8

# If you want more recent-only, uncomment and set date like "2026-01-01"
# FROM_DATE = "2026-01-01"
FROM_DATE = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# =========================
# HELPERS
# =========================

def sanitize_filename(text: str, max_len: int = 80) -> str:
    """
    Sanitize text to create a safe filename.
    
    Removes special characters, normalizes whitespace, and limits length.
    
    Args:
        text: Input text (typically article title)
        max_len: Maximum length for the filename
        
    Returns:
        Sanitized filename-safe string
    """
    text = text or "untitled"
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ", "_")
    return text[:max_len] if len(text) > max_len else text


def stable_id_from_url(url: str) -> str:
    """
    Generate a stable, short identifier from a URL using MD5 hash.
    
    Args:
        url: Article URL
        
    Returns:
        10-character hexadecimal hash string
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:10]


def fetch_newsapi_articles(query: str, page: int) -> List[dict]:
    """
    Fetch articles from NewsAPI for a given query and page number.
    
    Args:
        query: Search query string (supports OR operators)
        page: Page number (1-indexed)
        
    Returns:
        List of article dictionaries from NewsAPI, empty list on error
    """
    params = {
        "q": query,
        "language": LANGUAGE,
        "sortBy": SORT_BY,
        "pageSize": NEWSAPI_PAGE_SIZE,
        "page": page,
        "apiKey": NEWSAPI_KEY,
    }
    if FROM_DATE:
        params["from"] = FROM_DATE

    try:
        r = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        data = r.json()
    except RequestException as e:
        print(f"[NewsAPI] Request error: {e}")
        return []
    except json.JSONDecodeError:
        print("[NewsAPI] Bad JSON response.")
        return []

    if data.get("status") != "ok":
        print(f"[NewsAPI] Non-ok status: {data.get('status')}, message={data.get('message')}")
        return []

    return data.get("articles", []) or []


def extract_full_text_newspaper(url: str) -> Optional[str]:
    """Try to extract full article text from URL using newspaper3k."""
    try:
        a = Article(url, language=LANGUAGE)
        a.download()
        a.parse()
        txt = (a.text or "").strip()
        return txt if txt else None
    except Exception:
        return None


def text_quality_ok(extracted_text: str, api_content: str) -> bool:
    """
    Validate that extracted text meets quality thresholds.
    
    Checks minimum character count, word count, and compares against
    NewsAPI snippet length to ensure full text was extracted.
    
    Args:
        extracted_text: Full text extracted from article URL
        api_content: Content snippet from NewsAPI response
        
    Returns:
        True if text meets all quality criteria, False otherwise
    """
    extracted_text = (extracted_text or "").strip()
    api_content = (api_content or "").strip()

    if not extracted_text:
        return False

    # Basic length gates
    if len(extracted_text) < MIN_EXTRACTED_CHARS:
        return False

    words = extracted_text.split()
    if len(words) < MIN_EXTRACTED_WORDS:
        return False

    # Compare vs NewsAPI snippet length, if snippet exists
    if api_content:
        # NewsAPI "content" often ends with "… [+123 chars]" -> remove that
        api_clean = re.sub(r"\[\+\d+\schars\]$", "", api_content).strip()
        api_len = max(len(api_clean), 1)
        if (len(extracted_text) / api_len) < MIN_EXTRACTED_TO_NEWSAPI_RATIO:
            return False

    return True


def save_pdf(article: dict, topic: str, extracted_text: str, out_dir: str) -> str:
    """
    Create a formatted multi-page PDF from article data.
    
    Uses ReportLab's Platypus for automatic word wrapping and page breaks.
    Includes article metadata (title, source, published date, topic, URL) and
    full extracted text.
    
    Args:
        article: Article dictionary from NewsAPI
        topic: Topic category (e.g., "technology", "business")
        extracted_text: Full article text extracted from URL
        out_dir: Output directory for PDF files
        
    Returns:
        Path to the created PDF file
    """
    os.makedirs(out_dir, exist_ok=True)

    title = article.get("title") or "N/A"
    source = (article.get("source") or {}).get("name") or "N/A"
    published = article.get("publishedAt") or "N/A"
    description = article.get("description") or "N/A"
    url = article.get("url") or "N/A"

    # Unique filename: topic + stable url id + sanitized title
    url_id = stable_id_from_url(url) if url != "N/A" else "no_url"
    base = sanitize_filename(title)
    pdf_name = f"{topic}_{url_id}_{base}.pdf"
    pdf_path = os.path.join(out_dir, pdf_name)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LETTER,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"{title}", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Source:</b> {source}", styles["Normal"]))
    story.append(Paragraph(f"<b>Published:</b> {published}", styles["Normal"]))
    story.append(Paragraph(f"<b>Topic:</b> {topic}", styles["Normal"]))
    story.append(Paragraph(f"<b>URL:</b> {url}", styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Description:</b>", styles["Heading3"]))
    story.append(Paragraph(description, styles["BodyText"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Extracted Full Text:</b>", styles["Heading3"]))
    # Replace newlines with <br/> so Paragraph respects breaks
    safe_text = (extracted_text or "N/A").replace("\n", "<br/>")
    story.append(Paragraph(safe_text, styles["BodyText"]))

    doc.build(story)
    return pdf_path


# =========================
# MAIN
# =========================

def main():
    """
    Main execution function.
    
    Downloads articles from NewsAPI across all topics, extracts full text,
    validates quality, and saves as PDFs. Continues until target number
    of PDFs is reached or no more articles are available.
    """
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY" or not NEWSAPI_KEY.strip():
        raise ValueError("Set NEWSAPI_KEY to your real NewsAPI key.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen_urls = set()
    saved_count = 0

    print(f"Target PDFs: {TARGET_PDFS}")
    print(f"Output folder: {OUTPUT_DIR}\n")

    for topic, (query, desired_count) in TOPICS.items():
        if saved_count >= TARGET_PDFS:
            break

        print(f"\n=== Topic: {topic} | aiming for {desired_count} PDFs ===")
        topic_saved = 0

        for page in range(1, NEWSAPI_MAX_PAGES + 1):
            if saved_count >= TARGET_PDFS or topic_saved >= desired_count:
                break

            articles = fetch_newsapi_articles(query, page)
            if not articles:
                print(f"[{topic}] No more articles returned on page {page}.")
                break

            for art in articles:
                if saved_count >= TARGET_PDFS or topic_saved >= desired_count:
                    break

                url = art.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                api_content = (art.get("content") or "").strip()
                title = art.get("title") or "N/A"

                # Try full text extraction
                extracted = extract_full_text_newspaper(url)

                if not extracted:
                    # Skip saving if no extracted text
                    print(f"[SKIP] no extracted text | {topic} | {title}")
                    continue

                if not text_quality_ok(extracted, api_content):
                    print(f"[SKIP] extracted too short/weak | {topic} | {title}")
                    continue

                try:
                    save_pdf(art, topic, extracted, OUTPUT_DIR)
                    saved_count += 1
                    topic_saved += 1
                    print(f"[SAVE] ({saved_count}) {topic} | {title}")
                except Exception as e:
                    print(f"[ERROR] PDF save failed: {e}")

                time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

        print(f"--- Saved for {topic}: {topic_saved} PDFs ---")

    print(f"\nDONE. Total PDFs saved: {saved_count}")
    print(f"Folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

