"""
Cosine similarity search for semantic article retrieval.

This module provides functions for loading embeddings and metadata,
embedding search queries, and performing cosine similarity search
to find the most relevant articles.

Workflow:
1. Load pre-computed embeddings and metadata
2. Embed user query using SentenceTransformer
3. Compute cosine similarity between query and all documents
4. Return top-k most similar articles

Author: Semantic Search Engine Project
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_metadata(path="metadata.jsonl"):
    """
    Load article metadata from JSONL file.
    
    Args:
        path: Path to metadata JSONL file
        
    Returns:
        List of metadata dictionaries, one per article
    """
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def load_embeddings(path="embeddings.npy"):
    """
    Load pre-computed embeddings from NumPy file.
    
    Args:
        path: Path to .npy file containing embeddings
        
    Returns:
        NumPy array of embeddings (float32), shape (n_docs, embedding_dim)
    """
    return np.load(path).astype(np.float32)

def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a search query into vector space.
    
    Args:
        query: Search query string
        model_name: SentenceTransformer model name
        
    Returns:
        Normalized embedding vector (float32) for the query
    """
    model = SentenceTransformer(model_name)
    vec = model.encode([query], normalize_embeddings=True)
    return np.array(vec[0], dtype=np.float32)

def cosine_topk(query_vec: np.ndarray, doc_embs: np.ndarray, k: int = 10):
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

def search(query: str, k: int = 10, model_name: str = "all-MiniLM-L6-v2",
           emb_path="embeddings.npy", meta_path="metadata.jsonl"):
    """
    Perform semantic search and return top-k results.
    
    Complete search pipeline: loads data, embeds query, finds similar documents.
    
    Args:
        query: Search query string
        k: Number of top results to return
        model_name: SentenceTransformer model name
        emb_path: Path to embeddings file
        meta_path: Path to metadata file
        
    Returns:
        List of result dictionaries, each containing metadata plus 'score' and 'index'
    """
    embs = load_embeddings(emb_path)
    meta = load_metadata(meta_path)
    qvec = embed_query(query, model_name=model_name)
    idx, scores = cosine_topk(qvec, embs, k)

    # Combine metadata with similarity scores
    results = []
    for i, s in zip(idx, scores):
        r = meta[int(i)].copy()
        r["score"] = float(s)
        r["index"] = int(i)
        results.append(r)

    return results

if __name__ == "__main__":
    q = input("Query: ").strip()
    results = search(q, k=5)
    for rank, r in enumerate(results, 1):
        print(f"\n#{rank} score={r['score']:.4f}")
        print(r.get("title"))
        print(r.get("url"))
