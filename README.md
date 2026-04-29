# 🧠 RAG Chatbot (Retrieval-Augmented Generation)

A scalable and intelligent **RAG (Retrieval-Augmented Generation) chatbot** that combines the power of Large Language Models (LLMs) with a custom knowledge base to deliver accurate, context-aware responses.

---

## 🖼️ Demo / UI Preview

<img width="1847" height="1012" alt="RAG Chatbot UI" src="https://github.com/user-attachments/assets/5d9b3c20-fdb6-4ac3-b3a3-0d3a41c97a9f" />

---

## 🚀 Features

- 🔍 Semantic Search using vector embeddings  
- 🧠 Context-Aware Responses powered by LLMs  
- 📄 Document Ingestion (PDF, TXT, Markdown, etc.)  
- ⚡ FastAPI Backend for real-time interaction  
- 🗂️ Vector Database Integration (Pinecone / FAISS / Chroma)  
- 🔄 Scalable Architecture ready for production use  

---
🏗️ Architecture
User Query
   ↓
Embedding Model
   ↓
Vector Database (Top-K Retrieval)
   ↓
Context + Prompt
   ↓
LLM (OpenAI / LLaMA / etc.)
   ↓
Final Response

---

## 🛠️ Tech Stack

| Layer        | Technology              |
|-------------|------------------------|
| Backend     | FastAPI                |
| Language    | Python                 |
| LLM         | OpenAI                 |
| Embeddings  | Sentence Transformers  |
| Vector DB   | Pinecone               |
| Deployment  | Docker / AWS / Render  |

---

## 🛠️ Tech Stack

- **Backend**: FastAPI  
- **Language**: Python  
- **LLM**: OpenAI  
- **Embeddings**: Sentence Transformers  
- **Vector DB**: Pinecone  
- **Deployment**: Docker / AWS / Render  

---

## ⚙️ Setup

```bash
# Clone repo
git clone https://github.com/Ayushkumarsinghyogesh/RAG_super_chatbot.git
cd RAG_super_chatbot

# Create virtual env
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install deps
pip install -r requirements.txt


# RUN 
uvicorn app.main:app --reload

📥 Ingest Data
python3 -m app.utils.ingest

📂 Structure
RAG_super_chatbot/
├── app/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── utils/
├── data/
├── embeddings/
├── requirements.txt
└── README.md
