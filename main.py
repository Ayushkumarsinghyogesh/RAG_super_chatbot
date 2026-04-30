"""
main.py
-------
FastAPI application entry point.

Run locally:
  uvicorn main:app --reload --port 8000

Then open: http://localhost:8000/docs  ← Swagger UI to test your API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes.chat import router as chat_router

app = FastAPI(
    title="RAG Customer Support Bot",
    description="Answers customer questions using your product docs via OpenAI + Pinecone",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/chat", tags=["Chat"])


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}