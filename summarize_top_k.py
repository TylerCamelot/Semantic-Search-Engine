"""
Summarize top-k search results using AI or extractive methods.

This module performs semantic search and generates summaries for the top results.
Supports both OpenAI-based abstractive summarization and extractive fallback.

Workflow:
1. Perform semantic search to get top-k results
2. Load full article text for each result
3. Generate summaries (OpenAI if available, else extractive)
4. Return results with summaries attached

Author: Semantic Search Engine Project
"""

import os
import json
import re
from typing import Dict, List

from cosine_search import search

# Configuration
CLEAN_JSONL = "clean_articles.jsonl"  # File containing full article text

# Read OpenAI API key from environment variable
# Note: For Streamlit apps, use st.secrets in app.py instead
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")


def load_text_map(jsonl_path: str = CLEAN_JSONL) -> Dict[str, str]:
    """
    Load article text mapping from JSONL file.
    
    Creates a dictionary mapping PDF filenames to their full cleaned text.
    
    Args:
        jsonl_path: Path to JSONL file with article data
        
    Returns:
        Dictionary mapping filename -> full article text
    """
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fname = r.get("file")
            txt = (r.get("text") or "").strip()
            if fname and txt:
                mapping[fname] = txt
    return mapping


def extractive_fallback(text: str, max_sentences: int = 5) -> str:
    """
    Simple extractive summarizer: selects top sentences by word frequency.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if not sentences:
        return text[:900] + ("..." if len(text) > 900 else "")

    words = re.findall(r"[A-Za-z']{3,}", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    def score_sent(s: str) -> float:
        ws = re.findall(r"[A-Za-z']{3,}", s.lower())
        return sum(freq.get(w, 0) for w in ws) / (len(ws) + 1e-9)

    ranked = sorted(sentences, key=score_sent, reverse=True)
    chosen = ranked[:max_sentences]
    chosen_set = set(chosen)
    ordered = [s for s in sentences if s in chosen_set]
    return " ".join(ordered)


def summarize_openai(text: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        "Summarize the following news article in 5-7 bullet points. "
        "Be factual, concise, and avoid speculation.\n\n"
        f"ARTICLE:\n{text}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You summarize documents clearly and accurately."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def summarize_topk(query: str, k: int = 5, max_chars: int = 8000) -> List[dict]:
    """
    Search for top-k articles and generate summaries for each.
    
    Args:
        query: Search query string
        k: Number of top results to summarize
        max_chars: Maximum characters of article text to use for summarization
        
    Returns:
        List of result dictionaries with added 'summary' and 'summary_mode' fields
    """
    results = search(query, k=k)
    text_map = load_text_map(CLEAN_JSONL)

    out = []
    for r in results:
        fname = r.get("file")
        txt = text_map.get(fname, "")
        txt = txt[:max_chars] if txt else ""

        r2 = r.copy()
        if not txt:
            r2["summary"] = "(No text found for this document in clean_articles.jsonl.)"
            r2["summary_mode"] = "none"
        else:
            if OPENAI_API_KEY:
                try:
                    r2["summary"] = summarize_openai(txt)
                    r2["summary_mode"] = "openai"
                except Exception as e:
                    r2["summary"] = extractive_fallback(txt)
                    r2["summary_mode"] = f"fallback (openai error: {e})"
            else:
                r2["summary"] = extractive_fallback(txt)
                r2["summary_mode"] = "fallback (no OPENAI_API_KEY)"

        out.append(r2)

    return out


if __name__ == "__main__":
    q = input("Query: ").strip()
    k = int(input("Top-K: ").strip() or "5")

    results = summarize_topk(q, k=k)
    for i, r in enumerate(results, 1):
        print(f"\n#{i} score={r['score']:.4f} summary_mode={r['summary_mode']}")
        print(f"Title: {r.get('title')}")
        print(f"URL: {r.get('url')}\n")
        print(r.get("summary"))
