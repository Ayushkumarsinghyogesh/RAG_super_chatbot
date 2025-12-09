import json
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Init clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


def embed(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding


# ---------------------
# Query Pinecone
# ---------------------
def search(query):
    query_vec = embed(query)
    results = index.query(
        vector=query_vec,
        top_k=5,
        include_metadata=True
    )
    return results.matches


# ---------------------
# OpenAI Answer
# ---------------------
def generate_answer(context, question):
    prompt = f"""
    You are an internal company assistant. Answer only based on the given context.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Provide the best possible short and clear answer.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


# ---------------------
# Lambda Handler
# ---------------------
def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    query = body.get("query", "")

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "query field is required"})
        }

    matches = search(query)
    context_text = "\n".join([m.metadata.get("text", "") for m in matches])
    answer = generate_answer(context_text, query)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "query": query,
            "answer": answer,
            "source_context": context_text
        })
    }
