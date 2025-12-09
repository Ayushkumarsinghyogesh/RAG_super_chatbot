from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.vector_store import search_vector_store, embed_text
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

class QueryModel(BaseModel):
    query: str


@router.post("/")
async def chat(req: QueryModel):

    # Step 1: Create embedding for user query
    query_embedding = embed_text(req.query)

    # Step 2: Search vector DB
    matches = search_vector_store(query_embedding)

    # Build context
    context_text = "\n".join([m["metadata"]["text"] for m in matches])

    # Step 3: Use OpenAI for answer generation
    prompt = f"""
    You are an internal company assistant. Use the following context to answer:

    CONTEXT:
    {context_text}

    QUESTION:
    {req.query}

    Answer in a clear and short manner.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "query": req.query,
        "answer": response.choices[0].message.content,
        "context_used": matches
    }
