# Team Contributions

---

## Tyler Camelot

Built the article download pipeline in `download_articles.py`. That includes the NewsAPI integration (rate limiting, timeouts, user-agent), the filename helpers, and full-text extraction with newspaper3k. Also added the quality checks (min length, word count, snippet ratio), wrote the ReportLab PDF generation with metadata headers and word wrapping, and put together the main loop that walks topics and pages, dedupes URLs, and logs progress. On `app.py` helped with the search input, wiring the search call into the UI, and displaying result metadata and links. Maintained `requirements.txt`.

---

## Carson Pimental

Wrote `cosine_search.py`: loads the embeddings and metadata, embeds the query and runs L2 normalization, then does the cosine similarity and picks the top-k with `argpartition`. Handles threshold filtering and builds the result structure. Wrote `ARCHITECTURE.md`—overview, pipeline diagrams, how each piece fits together, data formats, algorithms, API notes, and the performance/scalability section.

---

## Sruthi Perikala

Owned `preprocess_pdf.py`: pulls text out of the PDFs with pypdf, fixes hyphenation and newlines, strips boilerplate, parses metadata from the headers, applies the word-count filter, and writes everything to JSONL and per-article TXT files. In `summarize_top_k.py` built the mapping from filenames to full text, hooked up OpenAI (GPT-4o-mini) for abstractive summaries, added the extractive fallback using word frequency and sentence scoring, and handled errors and the return format.

---

## Kevin Sohn

Built `create_embeddings.py`: loads the SentenceTransformer model, reads the clean articles, batch-encodes and L2-normalizes, then writes `embeddings.npy` and the aligned `metadata.jsonl`, with a progress bar. In `app.py` did the core logic—caching for the model, embeddings, metadata, and text map, the search flow and the 0.25 similarity threshold, session state, and the result list with expandable cards.

---

## Sherry Hsu

Handled the UI side of `app.py`: layout, custom CSS, and how the result cards look. Wired up AI title generation for untitled articles and the Summarize toggle/expanders. Wrote the README (overview, setup, API keys, how to run) and set up `.gitignore`.

---
