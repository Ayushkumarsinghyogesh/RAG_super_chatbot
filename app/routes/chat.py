"""
chat.py
-------
FastAPI route — RAG pipeline using:
  - sentence-transformers for embedding (FREE, local)
  - Groq for answer generation (FREE API)

THE RAG PIPELINE:
  1. User asks a question
  2. Embed question locally (sentence-transformers, no API cost)
  3. Search Pinecone for top matching chunks
  4. Send context + question to Groq (free LLM)
  5. Return grounded answer
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
from app.utils.vector_store import embed_text, search_vector_store

load_dotenv()

logger     = logging.getLogger(__name__)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
router     = APIRouter()

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)

class SourceChunk(BaseModel):
    id: str
    score: float
    text: str
    source: str

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer questions ONLY based on the provided context from our documentation.
If the answer is not in the context, say: "I don't have enough information about that. Please contact support."
Never make up information. Be concise and friendly."""

# ── MAIN ROUTE ────────────────────────────────────────────────────────────────
@router.post("/", response_model=ChatResponse)
async def chat(req: QueryRequest):
    # Step 1: Embed query locally (free, no API)
    try:
        query_embedding = embed_text(req.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Step 2: Search Pinecone
    try:
        matches = search_vector_store(query_embedding, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Pinecone error: {e}")
        raise HTTPException(status_code=502, detail="Vector search unavailable.")

    # Filter low-relevance results
    good_matches = [m for m in matches if m.score >= 0.30]

    if not good_matches:
        return ChatResponse(
            query=req.query,
            answer="I don't have enough information about that in our documentation. Please contact support.",
            sources=[],
        )

    # Step 3: Build context
    context_parts = []
    for i, m in enumerate(good_matches, 1):
        source = m.metadata.get("source", "unknown")
        text   = m.metadata.get("text", "")
        context_parts.append(f"[Source {i} - {source}]\n{text}")
    context_text = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate answer with Groq (free)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   # free Groq model, very fast
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"CONTEXT:\n{context_text}\n\nQUESTION:\n{req.query}"},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise HTTPException(status_code=502, detail="Answer generation unavailable.")

    # Step 5: Return
    sources = [
        SourceChunk(
            id=m.id,
            score=round(m.score, 4),
            text=m.metadata.get("text", ""),
            source=m.metadata.get("source", "unknown"),
        )
        for m in good_matches
    ]

    return ChatResponse(query=req.query, answer=answer, sources=sources)