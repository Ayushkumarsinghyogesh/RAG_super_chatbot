# 🧠 RAG Chatbot (Retrieval-Augmented Generation)

A scalable and intelligent **RAG (Retrieval-Augmented Generation) chatbot** that combines LLMs with a custom knowledge base to deliver accurate, context-aware answers.

---

## 🖼️ Demo / UI Preview

<img width="1847" height="1012" alt="RAG Chatbot UI" src="https://github.com/user-attachments/assets/5d9b3c20-fdb6-4ac3-b3a3-0d3a41c97a9f" />

---

## 🚀 Features

* 🔍 Semantic Search using vector embeddings
* 🧠 Context-aware responses using LLM
* 📄 Document ingestion (PDF, TXT, Markdown)
* ⚡ FastAPI backend
* 🗂️ Pinecone vector database
* 🔄 Scalable architecture

---

## 🏗️ Architecture

```
User Query
   ↓
Embedding (Sentence Transformers)
   ↓
Vector DB (Pinecone - Top K)
   ↓
Context + Prompt
   ↓
LLM (Groq - LLaMA / Mixtral)
   ↓
Final Response
```

---

## 🛠️ Tech Stack

| Layer      | Technology             |
| ---------- | ---------------------- |
| Backend    | FastAPI                |
| Language   | Python                 |
| LLM        | Groq (LLaMA / Mixtral) |
| Embeddings | Sentence Transformers  |
| Vector DB  | Pinecone               |
| Deployment | AWS Lambda / Render    |

---

## ⚙️ Setup

```bash
git clone https://github.com/Ayushkumarsinghyogesh/RAG_super_chatbot.git
cd RAG_super_chatbot

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload
```

---

## 📥 Ingest Data

```bash
python3 -m app.utils.ingest
```

---

## 📂 Project Structure

```
RAG_super_chatbot/
├── app/
│   ├── main.py
│   ├── routes/
│   ├── utils/
├── data/
├── requirements.txt
└── README.md
```

---

## 🎯 Use Cases

* Internal knowledge chatbot
* Document Q&A system
* AI-powered search
* Customer support automation

---
