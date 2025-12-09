import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# Create index if not exists
def setup_index():
    existing = pinecone_client.list_indexes().names()

    if PINECONE_INDEX not in existing:
        pinecone_client.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pinecone_client.Index(PINECONE_INDEX)


index = setup_index()


# ---- EMBEDDING ----
def embed_text(text: str):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return embedding.data[0].embedding


# ---- SEARCH ----
def search_vector_store(query_embedding, top_k=5):
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return res.matches
