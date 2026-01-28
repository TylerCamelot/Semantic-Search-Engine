# Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Technical Stack](#technical-stack)
6. [Algorithms & Models](#algorithms--models)
7. [Data Structures](#data-structures)
8. [API Integrations](#api-integrations)
9. [Performance Considerations](#performance-considerations)
10. [Scalability](#scalability)

## System Overview

The Semantic News Search Engine is a multi-stage NLP pipeline that transforms raw news articles into a searchable semantic space. The system follows a modular architecture where each component handles a specific stage of the pipeline.

### High-Level Architecture

```
┌─────────────────┐
│   NewsAPI       │
│   (External)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ download_       │
│ articles.py     │──► PDF Files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ preprocess_     │
│ pdf.py          │──► JSONL + TXT Files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ create_         │
│ embeddings.py   │──► Embeddings + Metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ cosine_search   │     │ summarize_top_k │
│ .py             │     │ .py             │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ app.py          │
│ (Streamlit UI)  │
└─────────────────┘
```

## Architecture Diagram

### Data Pipeline Flow

```
Raw Articles (NewsAPI)
    │
    ├─► Article URLs
    │
    ├─► Full Text Extraction (newspaper3k)
    │
    ├─► Quality Validation
    │   ├─► Min 1500 characters
    │   ├─► Min 250 words
    │   └─► 3x longer than API snippet
    │
    ├─► PDF Generation (ReportLab)
    │   ├─► Multi-page with word wrapping
    │   ├─► Metadata headers
    │   └─► Structured content
    │
    ├─► PDF Text Extraction (pypdf)
    │
    ├─► Text Cleaning
    │   ├─► Hyphenation fixes
    │   ├─► Boilerplate removal
    │   └─► Whitespace normalization
    │
    ├─► Metadata Parsing
    │   ├─► Title, Source, Published Date
    │   ├─► Topic, URL
    │   └─► Word count
    │
    ├─► Embedding Generation (SentenceTransformer)
    │   ├─► Model: all-MiniLM-L6-v2
    │   ├─► Batch processing (32)
    │   └─► L2 normalization
    │
    ├─► Semantic Search (Cosine Similarity)
    │   ├─► Query embedding
    │   ├─► Similarity computation
    │   └─► Top-k selection
    │
    └─► Summarization (OpenAI/Extractive)
        ├─► Abstractive (OpenAI GPT)
        └─► Extractive (Word frequency)
```

## Component Architecture

### 1. Article Download Module (`download_articles.py`)

**Purpose**: Fetch articles from NewsAPI and extract full text.

**Key Components**:
- **NewsAPI Client**: HTTP requests to NewsAPI v2 endpoint
- **Text Extractor**: Uses `newspaper3k` library for full-text extraction
- **Quality Validator**: Ensures extracted text meets quality thresholds
- **PDF Generator**: Creates formatted PDFs using ReportLab Platypus

**Technical Details**:
- **Rate Limiting**: 0.8s sleep between requests
- **Timeout**: 20 seconds per request
- **User Agent**: Custom browser user agent for compatibility
- **Error Handling**: Graceful degradation on extraction failures

**Data Flow**:
```
NewsAPI Response → Article URLs → Full Text Extraction → Quality Check → PDF Generation
```

**Quality Gates**:
- Minimum 1500 characters
- Minimum 250 words
- Extracted text must be 3x longer than NewsAPI snippet

### 2. PDF Preprocessing Module (`preprocess_pdf.py`)

**Purpose**: Extract and clean text from generated PDFs.

**Key Components**:
- **PDF Reader**: Uses `pypdf` (PyPDF2 successor)
- **Text Cleaner**: Regex-based normalization
- **Metadata Parser**: Extracts structured metadata from PDF headers
- **Quality Filter**: Filters articles below minimum word count

**Technical Details**:
- **Text Extraction**: Page-by-page extraction, concatenated
- **Cleaning Pipeline**:
  1. Fix hyphenated line breaks (`exam-\nple` → `example`)
  2. Normalize newlines and whitespace
  3. Remove boilerplate (subscribe, cookies, etc.)
  4. Strip header metadata blocks
- **Minimum Word Count**: 200 words

**Output Format**:
- **JSONL**: One JSON object per line with metadata + text
- **TXT Files**: Individual text files for each article

### 3. Embedding Generation Module (`create_embeddings.py`)

**Purpose**: Generate semantic embeddings for all articles.

**Key Components**:
- **SentenceTransformer Model**: `all-MiniLM-L6-v2`
- **Batch Processor**: Processes articles in batches of 32
- **Normalizer**: L2 normalization for cosine similarity

**Technical Details**:
- **Model Architecture**: 
  - Base: Microsoft's MiniLM-L6-v2
  - Embedding Dimension: 384
  - Max Sequence Length: 256 tokens
- **Normalization**: L2 normalization applied to all embeddings
- **Data Type**: float32 for memory efficiency
- **Progress Tracking**: Progress bar for batch processing

**Output Files**:
- `embeddings.npy`: NumPy array (n_articles × 384)
- `metadata.jsonl`: Metadata aligned with embedding rows

### 4. Search Module (`cosine_search.py`)

**Purpose**: Perform semantic search using cosine similarity.

**Key Components**:
- **Query Embedder**: Embeds user queries using same model
- **Similarity Computer**: Vectorized cosine similarity
- **Top-K Selector**: Efficient top-k selection using argpartition

**Algorithm**:
```python
# Normalize query vector
q_norm = q / ||q||

# Normalize document embeddings
docs_norm = docs / ||docs||_2

# Cosine similarity = dot product (when normalized)
scores = docs_norm @ q_norm

# Top-k selection (O(n log k) using argpartition)
top_k_indices = argpartition(-scores, k)[:k]
```

**Performance**:
- **Time Complexity**: O(n) for similarity, O(n log k) for top-k
- **Memory**: Efficient in-place operations
- **Speed**: <100ms for 1000 articles

### 5. Summarization Module (`summarize_top_k.py`)

**Purpose**: Generate summaries for search results.

**Key Components**:
- **OpenAI Client**: GPT-4o-mini for abstractive summarization
- **Extractive Fallback**: Word-frequency based sentence selection
- **Text Mapper**: Maps filenames to full article text

**Summarization Methods**:

**1. Abstractive (OpenAI)**:
- Model: GPT-4o-mini
- Temperature: 0.2 (low for consistency)
- Format: 5-7 bullet points
- Max tokens: Based on model limits

**2. Extractive (Fallback)**:
- Algorithm: Word frequency scoring
- Process:
  1. Calculate word frequencies in document
  2. Score sentences by average word frequency
  3. Select top 5 sentences
  4. Preserve original order

**Error Handling**:
- Falls back to extractive if OpenAI fails
- Handles missing text gracefully
- Returns summary mode indicator

### 6. Web Application (`app.py`)

**Purpose**: Interactive Streamlit-based search interface.

**Key Components**:
- **Streamlit Framework**: Web UI framework
- **Caching Layer**: `@st.cache_resource` and `@st.cache_data`
- **Search Engine**: Integrates cosine_search functionality
- **Title Generator**: AI-powered title generation for untitled articles

**Technical Details**:
- **Caching Strategy**:
  - Model: Cached as resource (loaded once)
  - Embeddings: Cached as data (reloads on file change)
  - Metadata: Cached as data
  - Text Map: Cached as data
- **State Management**: Streamlit's session state
- **UI Framework**: Custom CSS for styling

**Features**:
- Similarity threshold filtering (0.25)
- Real-time search
- Expandable result cards
- Topic-based emojis
- Gradient styling

## Data Flow

### Complete Pipeline

```
1. NewsAPI Articles (JSON)
   ↓
2. Full Text Extraction (newspaper3k)
   ↓
3. Quality Validation
   ↓
4. PDF Generation (ReportLab)
   ↓
5. PDF Storage (pdf_articles_fulltext/)
   ↓
6. PDF Text Extraction (pypdf)
   ↓
7. Text Cleaning & Normalization
   ↓
8. Metadata Extraction
   ↓
9. JSONL Output (clean_articles.jsonl)
   ↓
10. Embedding Generation (SentenceTransformer)
    ↓
11. Embeddings Storage (embeddings.npy)
    ↓
12. Metadata Storage (metadata.jsonl)
    ↓
13. Search Query (User Input)
    ↓
14. Query Embedding
    ↓
15. Cosine Similarity Computation
    ↓
16. Top-K Selection
    ↓
17. Result Display + Summarization
```

### Search Flow

```
User Query
    ↓
Query Embedding (SentenceTransformer)
    ↓
Load Document Embeddings (from disk)
    ↓
Normalize Vectors (L2)
    ↓
Compute Cosine Similarity (vectorized)
    ↓
Select Top-K (argpartition)
    ↓
Load Metadata (from JSONL)
    ↓
Filter by Threshold (≥ 0.25)
    ↓
Generate Summaries (if enabled)
    ↓
Display Results
```

## Technical Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Core language |
| NumPy | ≥1.24.0 | Numerical operations, embeddings |
| SentenceTransformers | ≥2.2.0 | Semantic embeddings |
| Streamlit | ≥1.28.0 | Web UI framework |
| OpenAI | ≥1.0.0 | AI summarization & titles |
| ReportLab | ≥4.0.0 | PDF generation |
| pypdf | ≥3.0.0 | PDF text extraction |
| newspaper3k | Latest | Article text extraction |
| requests | ≥2.31.0 | HTTP client |

### External Services

- **NewsAPI**: Article source (REST API)
- **OpenAI API**: Summarization and title generation (REST API)

## Algorithms & Models

### 1. Semantic Embedding Model

**Model**: `all-MiniLM-L6-v2`

**Architecture**:
- Base: DistilBERT (6 layers, 384 dimensions)
- Training: Knowledge distillation from larger models
- Output: 384-dimensional vectors
- Normalization: L2 normalized

**Why This Model?**
- Fast inference (~100ms for 1000 articles)
- Good quality for semantic similarity
- Small model size (~80MB)
- Balanced speed/quality trade-off

**Alternatives Considered**:
- `all-mpnet-base-v2`: Better quality, slower
- `paraphrase-MiniLM-L6-v2`: Similar performance
- `all-MiniLM-L12-v2`: Larger, slower, better quality

### 2. Cosine Similarity

**Formula**:
```
similarity = (A · B) / (||A|| × ||B||)
```

**Implementation**:
- Vectors are pre-normalized (L2 norm = 1)
- Cosine similarity = dot product
- Computed via matrix multiplication: `docs @ query`

**Complexity**:
- Time: O(n × d) where n = documents, d = dimensions
- Space: O(n × d) for embeddings

### 3. Top-K Selection

**Algorithm**: Partial sorting using `argpartition`

**Why argpartition?**
- Faster than full sort: O(n log k) vs O(n log n)
- Only needs top-k, not full ordering
- NumPy optimized implementation

**Process**:
1. Partition array around k-th element
2. Sort only the top-k partition
3. Return indices in sorted order

### 4. Extractive Summarization

**Algorithm**: Word frequency scoring

**Steps**:
1. Tokenize document into words (length ≥ 3)
2. Calculate word frequency
3. Score sentences: `sum(word_freq) / sentence_length`
4. Select top N sentences
5. Preserve original order

**Limitations**:
- No semantic understanding
- May miss important context
- Sensitive to word repetition

### 5. Text Cleaning Pipeline

**Operations** (in order):
1. Hyphenation fix: `(\w)-\n(\w)` → `\1\2`
2. Newline normalization: `\r\n` → `\n`
3. Multiple blank lines: `\n{3,}` → `\n\n`
4. Whitespace collapse: `[ \t]+` → ` `
5. Boilerplate removal: Filter lines matching patterns
6. Final trim: Remove leading/trailing whitespace

## Data Structures

### Embeddings

**Format**: NumPy array
**Shape**: `(n_articles, 384)`
**Dtype**: `float32`
**Storage**: `embeddings.npy`

**Memory**: ~1.5MB per 1000 articles (384 × 4 bytes × 1000)

### Metadata

**Format**: JSONL (JSON Lines)
**Structure**:
```json
{
  "file": "technology_abc123_article_title.pdf",
  "title": "Article Title",
  "source": "Source Name",
  "published": "2024-01-15T10:30:00Z",
  "topic": "technology",
  "url": "https://example.com/article",
  "word_count": 450
}
```

**Storage**: `metadata.jsonl` (one JSON object per line)

### Clean Articles

**Format**: JSONL
**Structure**:
```json
{
  "file": "filename.pdf",
  "word_count": 450,
  "title": "Article Title",
  "source": "Source Name",
  "published": "2024-01-15T10:30:00Z",
  "topic": "technology",
  "url": "https://example.com/article",
  "text": "Full article text here..."
}
```

**Storage**: `clean_articles.jsonl`

### Search Results

**Format**: List of dictionaries
**Structure**:
```python
[
  {
    "file": "filename.pdf",
    "title": "Article Title",
    "source": "Source Name",
    "published": "2024-01-15T10:30:00Z",
    "topic": "technology",
    "url": "https://example.com/article",
    "word_count": 450,
    "score": 0.8234,  # Cosine similarity score
    "index": 42       # Index in embeddings array
  },
  ...
]
```

## API Integrations

### NewsAPI

**Endpoint**: `https://newsapi.org/v2/everything`

**Parameters**:
- `q`: Search query (supports OR operators)
- `language`: "en"
- `sortBy`: "publishedAt"
- `pageSize`: 100 (max)
- `page`: Page number
- `apiKey`: API key

**Rate Limits**:
- Free tier: 100 requests/day
- Response: JSON with articles array

**Error Handling**:
- Rate limit detection
- JSON parsing errors
- Network timeouts (20s)

### OpenAI API

**Endpoints Used**:
- `chat.completions.create` (GPT-4o-mini)

**Summarization Request**:
```python
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You summarize documents..."},
    {"role": "user", "content": "Summarize: ..."}
  ],
  "temperature": 0.2
}
```

**Title Generation Request**:
```python
{
  "model": "gpt-4o-mini",
  "messages": [...],
  "temperature": 0.3,
  "max_tokens": 30
}
```

**Error Handling**:
- API key validation
- Rate limit handling
- Fallback to extractive methods
- Timeout handling

## Performance Considerations

### Embedding Generation

**Bottlenecks**:
- Model loading: ~2-3 seconds (one-time)
- Batch processing: ~100ms per batch of 32
- Total time: ~2-5 minutes for 1000 articles

**Optimizations**:
- Batch processing (32 articles at once)
- Float32 instead of float64 (2x memory savings)
- Progress bar for user feedback

### Search Performance

**Bottlenecks**:
- Embedding loading: ~50ms (cached after first load)
- Query embedding: ~50ms
- Similarity computation: ~10ms for 1000 articles
- Top-k selection: ~5ms

**Total**: <100ms per search query

**Optimizations**:
- Streamlit caching (embeddings loaded once)
- Vectorized NumPy operations
- Efficient top-k selection (argpartition)

### Memory Usage

**Components**:
- Embeddings: ~1.5MB per 1000 articles
- Metadata: ~500KB per 1000 articles
- Text map: ~50MB per 1000 articles (variable)
- Model: ~80MB (SentenceTransformer)

**Total**: ~132MB for 1000 articles

**Optimizations**:
- Float32 embeddings
- Lazy loading of text (only when needed)
- Streamlit caching reduces reloads

## Scalability

### Current Limitations

1. **Embeddings**: Loaded entirely into memory
2. **Search**: Linear scan through all embeddings
3. **Text Storage**: All text loaded into memory map

### Scaling Strategies

**For 10,000+ Articles**:

1. **Vector Database**:
   - Use FAISS, Pinecone, or Weaviate
   - Enables approximate nearest neighbor search
   - Reduces memory footprint

2. **Chunked Embeddings**:
   - Load embeddings in chunks
   - Process searches in batches
   - Use memory-mapped files

3. **Distributed Search**:
   - Shard embeddings across multiple servers
   - Parallel similarity computation
   - Aggregate results

4. **Caching Layer**:
   - Redis for frequently accessed articles
   - Cache popular queries
   - Reduce API calls

5. **Async Processing**:
   - Background embedding generation
   - Queue-based summarization
   - Non-blocking UI updates

### Recommended Architecture for Scale

```
┌─────────────┐
│  Load       │
│  Balancer   │
└──────┬──────┘
       │
       ├──► Streamlit App (UI)
       │
       ├──► Search Service (FastAPI)
       │    ├──► Vector DB (FAISS/Pinecone)
       │    └──► Cache (Redis)
       │
       └──► Processing Pipeline
            ├──► Message Queue (RabbitMQ/Kafka)
            ├──► Workers (Embedding Generation)
            └──► Storage (S3/PostgreSQL)
```

## Security Considerations

### API Key Management

- **Environment Variables**: Preferred method
- **Streamlit Secrets**: For deployment
- **Never Commit**: Keys excluded from version control

### Data Privacy

- **Article Content**: Stored locally
- **No User Data**: No tracking or analytics
- **OpenAI API**: Text sent to external service (consider for sensitive content)

### Input Validation

- **Query Sanitization**: Basic input validation
- **File Paths**: Validated before file operations
- **JSON Parsing**: Error handling for malformed data

## Testing Considerations

### Unit Tests Needed

1. **Text Cleaning**: Verify cleaning pipeline
2. **Similarity Computation**: Test cosine similarity accuracy
3. **Top-K Selection**: Verify correct ordering
4. **Metadata Parsing**: Test regex patterns

### Integration Tests Needed

1. **End-to-End Pipeline**: Full workflow test
2. **API Integration**: Mock NewsAPI/OpenAI responses
3. **Error Handling**: Test failure scenarios

### Performance Tests Needed

1. **Search Latency**: Measure query response time
2. **Memory Usage**: Profile memory consumption
3. **Concurrent Users**: Test Streamlit app under load

## Future Enhancements

1. **Multi-language Support**: Extend to non-English articles
2. **Advanced Filtering**: Date ranges, sources, topics
3. **Query Expansion**: Synonym-based query enhancement
4. **Result Ranking**: Combine similarity with recency/popularity
5. **User Feedback**: Click-through rate learning
6. **Export Functionality**: Save search results to CSV/PDF
7. **Batch Search**: Process multiple queries at once
8. **Visualizations**: Similarity score distributions, topic clusters

---

**Last Updated**: January 2026
**Version**: 1.0
