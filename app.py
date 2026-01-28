"""
Streamlit web application for semantic news article search.

This module provides an interactive web interface for searching news articles
using semantic similarity. Features include:
- Real-time semantic search with cosine similarity
- AI-powered article summarization (OpenAI or extractive fallback)
- Automatic title generation for untitled articles
- Similarity score filtering
- Modern, colorful UI with topic-based emojis

Workflow:
1. User enters search query
2. Query is embedded using SentenceTransformer
3. Cosine similarity search finds top-k articles
4. Results are filtered by similarity threshold
5. Summaries and titles are generated on-demand
6. Results displayed in interactive cards

Author: Semantic Search Engine Project
"""

import os
import json
import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
EMB_PATH = "embeddings.npy"
META_PATH = "metadata.jsonl"
CLEAN_JSONL = "clean_articles.jsonl"

MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
OPENAI_MODEL = os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")

# Function to get OpenAI API key (reads from Streamlit secrets or environment)
def get_openai_api_key():
    """
    Get OpenAI API key from Streamlit secrets or environment variable.
    
    Returns:
        API key string, empty if not found
    """
    try:
        # Try Streamlit secrets first (when running in Streamlit)
        return st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()
    except (AttributeError, FileNotFoundError, KeyError):
        # Fallback to environment variable if secrets not available
        return os.environ.get("OPENAI_API_KEY", "").strip()

# Initialize API key (will be set when Streamlit initializes)
OPENAI_API_KEY = ""

DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.25  # Minimum similarity score to show results

# Topic emoji mapping
TOPIC_EMOJIS = {
    "technology": "💻",
    "business": "💼",
    "politics": "🏛️",
    "world": "🌍",
    "science": "🔬",
    "health": "🏥",
    "sports": "⚽",
    "entertainment": "🎬",
    "": "📄"  # default
}


# -----------------------------
# DATA LOADERS (cached)
# -----------------------------
@st.cache_resource
def load_model():
    """
    Load and cache the SentenceTransformer model.
    
    Uses Streamlit's cache_resource to avoid reloading the model on every run.
    
    Returns:
        Loaded SentenceTransformer model instance
    """
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_embeddings():
    """
    Load and cache document embeddings from NumPy file.
    
    Returns:
        NumPy array of embeddings (float32), shape (n_docs, embedding_dim)
    """
    return np.load(EMB_PATH).astype(np.float32)

@st.cache_data
def load_metadata():
    """
    Load and cache article metadata from JSONL file.
    
    Returns:
        List of metadata dictionaries, one per article
    """
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

@st.cache_data
def load_text_map():
    """
    Load and cache article text mapping from JSONL file.
    
    Creates a dictionary mapping PDF filenames to their full cleaned text.
    Used for generating summaries and titles.
    
    Returns:
        Dictionary mapping filename -> full article text
    """
    mapping = {}
    with open(CLEAN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fname = r.get("file")
            txt = (r.get("text") or "").strip()
            if fname and txt:
                mapping[fname] = txt
    return mapping


# -----------------------------
# SEARCH (cosine similarity)
# -----------------------------
def cosine_topk(query_vec: np.ndarray, doc_embs: np.ndarray, k: int):
    """
    Find top-k documents using cosine similarity.
    
    Normalizes vectors and computes cosine similarity via dot product.
    Uses efficient argpartition for top-k selection when k < total docs.
    
    Args:
        query_vec: Query embedding vector (1D array)
        doc_embs: Document embeddings (2D array, shape: n_docs x embedding_dim)
        k: Number of top results to return
        
    Returns:
        Tuple of (indices, scores) for top-k documents, sorted by score descending
    """
    # Normalize query vector
    q = query_vec.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    # Normalize document embeddings
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12
    docs = doc_embs / norms

    # Cosine similarity = dot product (when normalized)
    scores = docs @ q

    # Efficient top-k selection
    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx_unsorted = np.argpartition(-scores, k)[:k]
        idx = idx_unsorted[np.argsort(-scores[idx_unsorted])]
    return idx, scores[idx]


def search(query: str, k: int):
    """
    Perform semantic search and return top-k results.
    
    Complete search pipeline: loads data, embeds query, finds similar documents.
    
    Args:
        query: Search query string
        k: Number of top results to return
        
    Returns:
        List of result dictionaries, each containing metadata plus 'score' and 'index'
    """
    model = load_model()
    embs = load_embeddings()
    meta = load_metadata()

    # Embed query and find top-k similar documents
    qvec = model.encode([query], normalize_embeddings=True)[0]
    idx, scores = cosine_topk(qvec, embs, k)

    # Combine metadata with similarity scores
    results = []
    for i, s in zip(idx, scores):
        r = meta[int(i)].copy()
        r["score"] = float(s)
        r["index"] = int(i)
        results.append(r)
    return results


# -----------------------------
# SUMMARIZATION (Option A)
# -----------------------------
def extractive_fallback(text: str, max_sentences: int = 5) -> str:
    """
    Generate extractive summary using word frequency scoring.
    
    Selects top sentences based on word frequency in the document.
    Falls back to first 900 characters if sentence extraction fails.
    
    Args:
        text: Article text to summarize
        max_sentences: Maximum number of sentences to include
        
    Returns:
        Extractive summary string
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    if not sentences:
        return text[:900] + ("..." if len(text) > 900 else "")

    # Calculate word frequencies
    words = re.findall(r"[A-Za-z']{3,}", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Score sentences by word frequency
    def score_sent(s: str) -> float:
        ws = re.findall(r"[A-Za-z']{3,}", s.lower())
        return sum(freq.get(w, 0) for w in ws) / (len(ws) + 1e-9)

    # Select top sentences while preserving order
    ranked = sorted(sentences, key=score_sent, reverse=True)
    chosen = ranked[:max_sentences]
    chosen_set = set(chosen)
    ordered = [s for s in sentences if s in chosen_set]
    return " ".join(ordered)


def summarize_openai(text: str) -> str:
    """
    Generate abstractive summary using OpenAI API.
    
    Uses GPT model to create concise bullet-point summary of the article.
    
    Args:
        text: Article text to summarize
        
    Returns:
        AI-generated summary string
        
    Raises:
        Exception: If OpenAI API call fails
    """
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


def summarize_text(text: str, max_chars: int = 8000) -> tuple[str, str]:
    """
    Generate summary using OpenAI or extractive fallback.
    
    Tries OpenAI first if API key is available, otherwise uses extractive method.
    
    Args:
        text: Article text to summarize
        max_chars: Maximum characters of text to use for summarization
        
    Returns:
        Tuple of (summary_text, summary_mode) where mode indicates the method used
    """
    text_clip = (text or "")[:max_chars]
    if not text_clip:
        return "(No text available.)", "none"

    if OPENAI_API_KEY:
        try:
            return summarize_openai(text_clip), "openai"
        except Exception as e:
            return extractive_fallback(text_clip), f"fallback (openai error: {e})"
    else:
        return extractive_fallback(text_clip), "fallback (no OPENAI_API_KEY)"


def generate_title_openai(text: str, max_chars: int = 2000) -> str:
    """
    Generate article title using OpenAI API.
    
    Creates a concise, engaging title (max 10 words) based on article content.
    
    Args:
        text: Article text
        max_chars: Maximum characters of text to use
        
    Returns:
        Generated title string, or "Untitled" on error
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    text_clip = (text or "")[:max_chars]
    if not text_clip:
        return "Untitled"

    prompt = (
        "Generate a concise, engaging news article title (maximum 10 words) "
        "based on the following article text. Return only the title, nothing else.\n\n"
        f"ARTICLE TEXT:\n{text_clip}"
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You generate concise, accurate news article titles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=30,
        )
        title = resp.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip('"\'')
        return title if title else "Untitled"
    except Exception as e:
        return "Untitled"


def generate_title_fallback(text: str) -> str:
    """
    Generate title using extractive method (first sentence or key phrases).
    
    Falls back to this method when OpenAI is unavailable.
    Uses first sentence if reasonable length, otherwise truncates intelligently.
    
    Args:
        text: Article text
        
    Returns:
        Generated title string
    """
    if not text:
        return "Untitled"
    
    # Try to extract first sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if sentences:
        first_sent = sentences[0].strip()
        # Limit to reasonable length
        if len(first_sent) > 80:
            # Try to find a good break point
            words = first_sent.split()
            title_words = []
            for word in words:
                if len(' '.join(title_words + [word])) <= 80:
                    title_words.append(word)
                else:
                    break
            return ' '.join(title_words) + "..."
        return first_sent[:80]
    
    # Fallback: use first 60 characters
    return text[:60].strip() + "..."


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="Semantic Search Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize OpenAI API key after Streamlit is initialized
OPENAI_API_KEY = get_openai_api_key()

# Custom CSS with vibrant colors and modern design
st.markdown("""
    <style>
    /* Main layout */
    .main > div {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Title centering with gradient */
    h1 {
        text-align: center;
        margin-bottom: 1rem;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Search input */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
    }
    
    /* Search button with gradient */
    .stButton > button {
        width: 100%;
        font-weight: 600;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Headings */
    h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        color: #667eea;
    }
    
    /* Result cards with gradient backgrounds */
    .result-item-wrapper {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        padding: 1.25rem !important;
        border-radius: 0.75rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .result-item-wrapper:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #764ba2;
        font-weight: 600;
    }
    
    /* Expanders */
    .stExpander {
        margin-bottom: 0.5rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        padding-top: 1rem;
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Badges and metadata */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    
    /* Remove default dividers */
    hr {
        display: none;
    }
    
    /* Links */
    a {
        color: #667eea;
        text-decoration: none;
    }
    a:hover {
        color: #764ba2;
        text-decoration: underline;
    }
    
    /* Status badges */
    .stSuccess {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stInfo {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header - Centered
st.title("Semantic News Search")

# Status indicator
status_col1, status_col2, status_col3 = st.columns([1, 1, 1])
with status_col2:
    if OPENAI_API_KEY:
        st.success("✓ AI Summaries")
    else:
        st.info("📝 Extractive Only")

# Basic checks
missing = []
for p in [EMB_PATH, META_PATH, CLEAN_JSONL]:
    if not os.path.exists(p):
        missing.append(p)

if missing:
    st.error("Missing required files:\n" + "\n".join(missing))
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    k = st.slider("Top-K results", 1, 20, DEFAULT_K, help="Number of results to return")
    do_summarize = st.checkbox("Generate summaries", value=True)
    max_chars = st.slider("Max chars per doc", 2000, 12000, 8000, step=1000)
    st.markdown("---")
    st.caption(f"Model: `{MODEL_NAME}`")

# Search interface
search_col1, search_col2 = st.columns([5, 1])
with search_col1:
    query = st.text_input("", value="", placeholder="Enter your search query...", label_visibility="collapsed")
with search_col2:
    search_btn = st.button("🔎 Search", use_container_width=True)

if search_btn:
    # Check if query is empty
    if not query or not query.strip():
        st.warning("⚠️ Please enter a search query to find articles.")
    else:
        with st.spinner("Searching..."):
            results = search(query.strip(), k=k)
            text_map = load_text_map()

        if results:
            # Check if top result meets similarity threshold
            top_score = results[0].get("score", 0.0)
            
            if top_score < SIMILARITY_THRESHOLD:
                st.info(f"🔍 No similar results found for this search. The highest similarity score ({top_score:.3f}) is below the threshold ({SIMILARITY_THRESHOLD}). Try refining your query.")
            else:
                # Filter results that meet the threshold
                filtered_results = [r for r in results if r.get("score", 0.0) >= SIMILARITY_THRESHOLD]
                
                if filtered_results:
                    st.markdown(f"### Found {len(filtered_results)} results")
                    
                    for rank, r in enumerate(filtered_results, 1):
                        title = r.get("title") or ""
                        url = r.get("url") or ""
                        topic = r.get("topic") or ""
                        source = r.get("source") or ""
                        published = r.get("published") or ""
                        score = r.get("score", 0.0)
                        fname = r.get("file")
                        
                        # Get full text for title generation if needed
                        full_text = text_map.get(fname, "")
                        
                        # Generate title if missing or "Untitled"
                        if not title or title.lower() == "untitled" or title == "N/A":
                            if OPENAI_API_KEY and full_text:
                                with st.spinner(f"Generating title for result {rank}..."):
                                    title = generate_title_openai(full_text)
                            elif full_text:
                                title = generate_title_fallback(full_text)
                            else:
                                title = "Untitled"
                        
                        # Get emoji for topic
                        topic_emoji = TOPIC_EMOJIS.get(topic.lower(), TOPIC_EMOJIS.get(""))
                        
                        # Compact result card with grey background
                        st.markdown(f'<div class="result-item-wrapper" id="result-{rank}">', unsafe_allow_html=True)
                        
                        # Header row with title and score
                        header_col1, header_col2 = st.columns([4, 1])
                        with header_col1:
                            st.markdown(f"**{topic_emoji} #{rank}** {title}")
                        with header_col2:
                            st.metric("Similarity", f"{score:.3f}", delta=None)
                        
                        # Metadata badges
                        if source or topic or published:
                            meta_cols = st.columns([2, 2, 3])
                            with meta_cols[0]:
                                if source:
                                    st.markdown(f"📰 **{source}**")
                            with meta_cols[1]:
                                if topic:
                                    st.markdown(f"🏷️ **{topic}**")
                            with meta_cols[2]:
                                if published:
                                    st.markdown(f"📅 {published[:10] if len(published) > 10 else published}")
                        
                        # Summary section
                        if do_summarize and full_text:
                            with st.expander("📝 Summary", expanded=(rank <= 2)):
                                summary, mode = summarize_text(full_text, max_chars=max_chars)
                                st.markdown(summary)
                                st.caption(f"Mode: {mode}")
                        
                        # URL and full text
                        if url:
                            st.markdown(f"🔗 [{url}]({url})")
                        
                        # Optional full text
                        if full_text:
                            with st.expander("📄 Full Text", expanded=False):
                                st.text_area("", value=full_text[:6000], height=150, label_visibility="collapsed", key=f"text_{rank}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No results found that meet the similarity threshold.")
        else:
            st.warning("No results found.")
