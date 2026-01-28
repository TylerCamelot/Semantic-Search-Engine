"""
Create embeddings for articles using SentenceTransformer.
Loads articles from JSONL, generates embeddings, and saves them.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_JSONL = "clean_articles.jsonl"
OUTPUT_EMBEDDINGS = "embeddings.npy"
OUTPUT_METADATA = "metadata.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"  # fast + solid quality
BATCH_SIZE = 32


# =========================
# LOADING
# =========================

def load_corpus(jsonl_path: str):
    """
    Load articles from JSONL file and extract texts.
    
    Args:
        jsonl_path: Path to JSONL file with articles
    
    Returns:
        Tuple of (records list, texts list)
    """
    records = []
    texts = []
    
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    print(f"Loading articles from {jsonl_path}...")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                text = record.get("text", "").strip()
                
                if not text:
                    print(f"Warning: Skipping record {line_num} - no text content")
                    continue
                
                records.append(record)
                texts.append(text)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(records)} articles with text content.")
    return records, texts


# =========================
# EMBEDDING GENERATION
# =========================

def create_embeddings(texts, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    """
    Generate embeddings for texts using SentenceTransformer.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for encoding
    
    Returns:
        numpy array of embeddings
    """
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # Convert to float32 for efficiency
    embeddings = np.array(embeddings, dtype=np.float32)
    
    return embeddings


# =========================
# SAVING
# =========================

def save_embeddings(embeddings, output_path=OUTPUT_EMBEDDINGS):
    """Save embeddings to .npy file."""
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")


def save_metadata(records, output_path=OUTPUT_METADATA):
    """
    Save metadata aligned with embeddings row order.
    
    Args:
        records: List of article records
        output_path: Path to output JSONL file
    """
    with open(output_path, "w", encoding="utf-8") as out:
        for r in records:
            metadata = {
                "file": r.get("file"),
                "title": r.get("title"),
                "source": r.get("source"),
                "published": r.get("published"),
                "topic": r.get("topic"),
                "url": r.get("url"),
                "word_count": r.get("word_count"),
            }
            out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    
    print(f"Saved metadata to {output_path}")


# =========================
# MAIN
# =========================

def main():
    """Main function to create embeddings."""
    print("=" * 60)
    print("Creating Embeddings for Semantic Search")
    print("=" * 60)
    print()
    
    # Load corpus
    try:
        records, texts = load_corpus(INPUT_JSONL)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nPlease run preprocess_pdf.py first to create {INPUT_JSONL}")
        return
    
    if not records or not texts:
        print("Error: No articles found in corpus.")
        return
    
    # Generate embeddings
    embeddings = create_embeddings(texts, MODEL_NAME, BATCH_SIZE)
    
    # Save results
    save_embeddings(embeddings, OUTPUT_EMBEDDINGS)
    save_metadata(records, OUTPUT_METADATA)
    
    print("\n" + "=" * 60)
    print("✓ Embeddings created successfully!")
    print("=" * 60)
    print(f"  Embeddings: {OUTPUT_EMBEDDINGS}")
    print(f"  Metadata:   {OUTPUT_METADATA}")
    print(f"  Shape:      {embeddings.shape}")
    print(f"  Model:      {MODEL_NAME}")


if __name__ == "__main__":
    main()
