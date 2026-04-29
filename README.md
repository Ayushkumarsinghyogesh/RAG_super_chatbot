🧠 RAG Chatbot (Retrieval-Augmented Generation)

A scalable and intelligent RAG (Retrieval-Augmented Generation) chatbot that combines the power of Large Language Models (LLMs) with a custom knowledge base to deliver accurate, context-aware responses.

🖼️ Demo / UI Preview
<img width="1847" height="1012" alt="RAG Chatbot UI" src="https://github.com/user-attachments/assets/5d9b3c20-fdb6-4ac3-b3a3-0d3a41c97a9f" />
🚀 Features
🔍 Semantic Search using vector embeddings
🧠 Context-Aware Responses powered by LLMs
📄 Document Ingestion (PDF, TXT, Markdown, etc.)
⚡ FastAPI Backend for real-time interaction
🗂️ Vector Database Integration (Pinecone / FAISS / Chroma)
🔄 Scalable Architecture ready for production use
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
🛠️ Tech Stack
Layer	Technology
Backend	FastAPI
Language	Python
LLM	OpenAI
Embeddings	Sentence Transformers
Vector DB	Pinecone
Deployment	Docker / AWS / Render
📦 Installation
1. Clone the Repository
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows
3. Install Dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_environment
VECTOR_DB_INDEX=your_index_name
▶️ Run the Application
uvicorn app.main:app --reload
Access the app:
http://127.0.0.1:8000
📂 Project Structure
rag-chatbot/
│── app/
│   ├── main.py          # FastAPI entry point
│   ├── routes/          # API routes
│   ├── services/        # Business logic (RAG pipeline)
│   ├── utils/           # Helper functions
│
│── data/                # Raw documents
│── embeddings/          # Vector storage (local if used)
│── requirements.txt
│── README.md
📥 Document Ingestion

To ingest documents into the vector database:

python ingest.py --path ./data
Supported formats:
PDF
TXT
Markdown
