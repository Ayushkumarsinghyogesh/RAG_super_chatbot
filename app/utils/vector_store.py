"""
vector_store.py
---------------
UPDATED: Uses sentence-transformers for embeddings (FREE, runs locally)
instead of OpenAI embeddings (paid).

WHY sentence-transformers?
  - Completely free, runs on your machine
  - No API calls needed for embedding
  - 'all-MiniLM-L6-v2' model → 384 dimensions, fast, good quality
  - Downloaded once (~90MB), then cached locally forever

Groq is used only for CHAT (answer generation), not embeddings.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── ENV VARS ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "rag-chat-bot-index")

if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY is missing from .env")

# ── EMBEDDING MODEL (local, free) ─────────────────────────────────────────────
# Downloaded once on first run (~90MB), cached at ~/.cache/huggingface/
# 384 dimensions, very fast, great for semantic search
print("[Embeddings] Loading local sentence-transformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Embeddings] Model loaded!")

# ── PINECONE CLIENT ───────────────────────────────────────────────────────────
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


# ── INDEX SETUP ───────────────────────────────────────────────────────────────
def setup_index():
    """
    Create Pinecone index if not exists, then return it.

    dimension=384   → matches all-MiniLM-L6-v2 output (NOT 1536 anymore)
    metric="cosine" → best for semantic similarity
    """
    existing_names = [idx["name"] for idx in pinecone_client.list_indexes()]

    if PINECONE_INDEX not in existing_names:
        print(f"[Pinecone] Creating index '{PINECONE_INDEX}'...")
        pinecone_client.create_index(
            name=PINECONE_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[Pinecone] Index '{PINECONE_INDEX}' created!")
    else:
        print(f"[Pinecone] Index '{PINECONE_INDEX}' already exists.")

    return pinecone_client.Index(PINECONE_INDEX)


index = setup_index()


# ── EMBED TEXT ────────────────────────────────────────────────────────────────
def embed_text(text):
    """
    Convert text to a 384-dim vector using local sentence-transformers.
    No API call, no cost, runs on CPU in milliseconds.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed an empty string.")
    return embedding_model.encode(text).tolist()


# ── SEARCH ────────────────────────────────────────────────────────────────────
def search_vector_store(query_embedding, top_k=5):
    """
    Find top_k most similar chunks in Pinecone for a given query vector.
    """
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    return results.matches