# Semantic News Search Engine

A powerful semantic search engine for news articles that uses advanced NLP techniques to find relevant articles based on meaning rather than just keywords. Built with SentenceTransformers, Streamlit, and OpenAI.

## 🎯 Project Overview

This project implements a complete semantic search pipeline for news articles:

1. **Article Collection**: Downloads articles from NewsAPI across 8 topics (technology, business, politics, world, science, health, sports, entertainment)
2. **Text Extraction**: Extracts full article text from URLs using newspaper3k
3. **PDF Generation**: Creates formatted PDFs with proper word wrapping and metadata
4. **Text Preprocessing**: Cleans and normalizes text, removes boilerplate
5. **Embedding Generation**: Creates semantic embeddings using SentenceTransformer
6. **Semantic Search**: Performs cosine similarity search to find relevant articles
7. **AI Summarization**: Generates summaries using OpenAI or extractive methods
8. **Web Interface**: Interactive Streamlit app with modern UI

## ✨ Features

- **Semantic Search**: Find articles by meaning, not just keywords
- **AI-Powered Summaries**: Automatic article summarization (OpenAI or extractive fallback)
- **Title Generation**: AI-generated titles for untitled articles
- **Similarity Filtering**: Only shows results above similarity threshold (0.25)
- **Topic Categorization**: Visual emojis for different article topics
- **Modern UI**: Colorful, responsive Streamlit interface
- **Quality Filtering**: Ensures articles meet minimum quality standards

## 📋 Prerequisites

- Python 3.8 or higher
- NewsAPI key (free tier available at [newsapi.org](https://newsapi.org))
- OpenAI API key (optional, for AI summaries and title generation)

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Semantic_Search_Engine
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

#### NewsAPI Key (Required)

Edit `download_articles.py` and set your NewsAPI key:

```python
NEWSAPI_KEY = "your-newsapi-key-here"
```

Or set it as an environment variable:

```bash
# Windows PowerShell
$env:NEWSAPI_KEY="your-newsapi-key-here"

# Linux/Mac
export NEWSAPI_KEY="your-newsapi-key-here"
```

#### OpenAI API Key (Optional)

For AI-powered summaries and title generation, set your OpenAI API key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-openai-key-here"
```

Or create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-openai-key-here"
```

### 4. Download Articles

```bash
python download_articles.py
```

This will:
- Download articles from NewsAPI across 8 topics
- Extract full text from article URLs
- Validate text quality
- Save articles as PDFs in `pdf_articles_fulltext/`
- Target: 1000 articles (125 per topic)

### 5. Preprocess PDFs

```bash
python preprocess_pdf.py
```

This will:
- Extract text from all PDFs
- Clean and normalize text
- Parse metadata (title, source, published date, topic, URL)
- Filter articles by minimum word count (200 words)
- Save cleaned articles to `clean_articles.jsonl` and `clean_txt/`

### 6. Generate Embeddings

```bash
python create_embeddings.py
```

This will:
- Load cleaned articles from `clean_articles.jsonl`
- Generate embeddings using SentenceTransformer (`all-MiniLM-L6-v2`)
- Save embeddings to `embeddings.npy`
- Save metadata to `metadata.jsonl`

### 7. Run the Web Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📖 Usage Examples

### Command-Line Search

```bash
python cosine_search.py
```

Example:
```
Query: AI regulation in Europe
```

Output:
```
#1  score=0.8234
Title: European Union Announces New AI Regulation Framework
URL: https://example.com/article1

#2  score=0.7891
Title: How Europe is Leading AI Governance
URL: https://example.com/article2
...
```

### Summarize Top-K Results

```bash
python summarize_top_k.py
```

Example:
```
Query: climate change solutions
Top-K: 5
```

Output:
```
#1 score=0.8567 summary_mode=openai
Title: Renewable Energy Breakthroughs in 2024
URL: https://example.com/article1

• Solar panel efficiency reaches new record high
• Wind energy costs continue to decline
• Battery storage technology advances rapidly
...
```

### Web Interface

1. **Start the app**:
   ```bash
   streamlit run app.py
   ```

2. **Search for articles**:
   - Enter your query in the search box
   - Click "Search" or press Enter
   - View results with similarity scores

3. **Customize settings** (sidebar):
   - Adjust Top-K results (1-20)
   - Toggle summaries on/off
   - Set max characters per document

4. **Explore results**:
   - View article metadata (source, topic, published date)
   - Read AI-generated summaries
   - Access full article text
   - Click URLs to read original articles

## 📁 Project Structure

```
Semantic_Search_Engine/
│
├── download_articles.py      # Download articles from NewsAPI and save as PDFs
├── preprocess_pdf.py         # Extract and clean text from PDFs
├── create_embeddings.py      # Generate semantic embeddings
├── cosine_search.py          # Cosine similarity search functions
├── summarize_top_k.py        # Summarize top-k search results
├── app.py                    # Streamlit web application
│
├── requirements.txt           # Python dependencies
├── README.md                 # This file
│
├── pdf_articles_fulltext/    # Generated PDF files (1000 articles)
├── clean_txt/                # Individual cleaned text files
│
├── clean_articles.jsonl      # Cleaned articles in JSONL format
├── embeddings.npy            # Pre-computed embeddings (numpy array)
├── metadata.jsonl            # Article metadata aligned with embeddings
│
└── .streamlit/
    └── secrets.toml          # Streamlit secrets (API keys)
```

## 🔧 Configuration

### Similarity Threshold

In `app.py`, adjust the similarity threshold:

```python
SIMILARITY_THRESHOLD = 0.25  # Minimum similarity score to show results
```

### Topics and Queries

In `download_articles.py`, customize topics:

```python
TOPICS = {
    "technology": ('technology OR tech OR AI OR "artificial intelligence"', 125),
    "business": ('business OR economy OR finance OR markets', 125),
    # ... add more topics
}
```

### Embedding Model

Change the embedding model in `create_embeddings.py` or `app.py`:

```python
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and efficient
# Alternatives: "all-mpnet-base-v2" (better quality, slower)
```

## 🎨 Features Explained

### Semantic Search

Unlike keyword search, semantic search understands meaning:
- **Keyword search**: "AI" only finds articles with the word "AI"
- **Semantic search**: "AI" finds articles about artificial intelligence, machine learning, neural networks, etc.

### Similarity Scoring

- Scores range from 0.0 to 1.0
- Higher scores = more relevant
- Threshold of 0.25 filters out irrelevant results

### AI Summarization

Two modes:
1. **OpenAI** (if API key available): Abstractive summaries with bullet points
2. **Extractive**: Word-frequency based sentence selection

### Title Generation

Automatically generates titles for untitled articles:
- Uses OpenAI if available
- Falls back to first sentence extraction

## 🐛 Troubleshooting

### "Missing required files" Error

Make sure you've run the pipeline in order:
1. `download_articles.py` → creates PDFs
2. `preprocess_pdf.py` → creates `clean_articles.jsonl`
3. `create_embeddings.py` → creates `embeddings.npy` and `metadata.jsonl`

### NewsAPI Rate Limits

Free tier allows 100 requests/day. The script includes:
- Rate limiting (sleep between requests)
- Error handling for rate limit responses

### OpenAI API Errors

If OpenAI API fails:
- Summaries fall back to extractive method
- Title generation falls back to first sentence
- Check your API key and account balance

### Memory Issues

For large datasets:
- Reduce `TARGET_PDFS` in `download_articles.py`
- Use smaller batch size in `create_embeddings.py`
- Process articles in smaller batches

## 📊 Performance

- **Embedding Generation**: ~1000 articles in 2-5 minutes
- **Search Speed**: <100ms for 1000 articles
- **Summary Generation**: 2-5 seconds per article (OpenAI)

## 🔐 Security Notes

- Never commit API keys to version control
- Use environment variables or `.streamlit/secrets.toml`
- The `.streamlit/secrets.toml` file is gitignored by default

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions, please open an issue on GitHub.

---
