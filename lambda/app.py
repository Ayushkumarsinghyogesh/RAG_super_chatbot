"""
lambda/app.py
-------------
AWS Lambda handler — updated to use:
  - sentence-transformers for embedding (free, local)
  - Groq for answer generation (free API)
"""

import json
import os
import logging
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "rag-chatbot")

# Init once per container (reused on warm Lambda calls)
groq_client     = Groq(api_key=GROQ_API_KEY)
pc              = Pinecone(api_key=PINECONE_API_KEY)
index           = pc.Index(PINECONE_INDEX)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer questions ONLY based on the provided context.
If the answer is not in the context, say you don't have that information.
Never make up information. Be concise and friendly."""

def embed(text):
    return embedding_model.encode(text.strip()).tolist()

def search_pinecone(query, top_k=5):
    vec     = embed(query)
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return [m for m in results.matches if m.score >= 0.30]

def generate_answer(context, question):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

def make_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }

def lambda_handler(event, context):
    logger.info(f"Event: {json.dumps(event)}")

    if event.get("httpMethod") == "OPTIONS":
        return make_response(200, {"message": "OK"})

    try:
        raw_body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        body = json.loads(raw_body)
    except Exception:
        return make_response(400, {"error": "Invalid JSON in request body."})

    query = body.get("query", "").strip()
    if not query:
        return make_response(400, {"error": "'query' field is required."})

    try:
        matches = search_pinecone(query)

        if not matches:
            return make_response(200, {
                "query": query,
                "answer": "I don't have enough information. Please contact support.",
                "sources": [],
            })

        context_parts = [
            f"[Source {i} - {m.metadata.get('source', 'unknown')}]\n{m.metadata.get('text', '')}"
            for i, m in enumerate(matches, 1)
        ]
        context_text = "\n\n---\n\n".join(context_parts)
        answer       = generate_answer(context_text, query)

        sources = [
            {"id": m.id, "score": round(m.score, 4),
             "text": m.metadata.get("text", ""), "source": m.metadata.get("source", "")}
            for m in matches
        ]

        return make_response(200, {"query": query, "answer": answer, "sources": sources})

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return make_response(500, {"error": "Internal server error."})