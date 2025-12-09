from fastapi import FastAPI
from app.routes.chat import router as chat_router

app = FastAPI(
    title="RAG Chatbot",
    description="A chatbot using Pinecone + OpenAI embeddings",
    version="1.0.0"
)

app.include_router(chat_router, prefix="/chat", tags=["Chat"])
